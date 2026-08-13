"""
rag.py
======
Standalone RAG pipeline (no FastAPI, no agent/tool calls).

Architecture:
  - Document loading : PyPDFDirectoryLoader over all PDFs in PDF_DIR
  - Chunking          : RecursiveCharacterTextSplitter (500 chars, 50 overlap)
  - Embeddings        : OpenAI text-embedding-3-small
  - Vector store      : ChromaDB, persisted locally at ./chroma_db
  - Retriever         : EnsembleRetriever = 70% Chroma (dense) + 30% BM25 (keyword)
  - LLM               : ChatOpenAI (gpt-4o-mini)
  - Tracing           : DeepEval @observe + update_current_trace (white box)
                         retrieved chunks are pushed into the trace as
                         retrieval_context so Contextual Precision/Recall
                         metrics in rag_eval.py have something to grade.

Run directly (`python rag.py`) to ask ONE question and get ONE answer
(no chat loop) — this mirrors how rag_eval.py calls rag_agent().
"""

import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from deepeval.tracing import observe
from deepeval.tracing.context import update_current_trace


PDF_DIR = os.path.expanduser("/Users/charithsaibanda/Downloads/Environment-Setup_2/ragg/Fin-pdfs")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DENSE_WEIGHT = 0.7   # Chroma (semantic) — weighted higher
BM25_WEIGHT = 0.3    # BM25 (keyword)
TOP_K = 3

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Clean text extracted from PDFs — some PDFs contain stylized Unicode
# characters (e.g. mathematical alphanumeric symbols in headers/fonts) that
# pypdf can extract as broken surrogate pairs. These crash when Chroma tries
# to UTF-8 encode them for storage, so we strip anything invalid here.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


# Step 1: Load PDFs from PDF_DIR and split into chunks

def load_and_chunk_documents(pdf_dir: str):
    loader = PyPDFDirectoryLoader(pdf_dir)
    all_docs = loader.load()

    if not all_docs:
        print(f"[WARNING] No documents were loaded from {pdf_dir}")
        return []

    # Clean surrogate/invalid characters out of each page's text before chunking.
    for doc in all_docs:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[INFO] Loaded {len(all_docs)} page(s) -> {len(chunks)} chunks")
    return chunks


# Step 2: Build Chroma (persisted) + BM25 -> EnsembleRetriever

def build_retriever():
    chunks = load_and_chunk_documents(PDF_DIR)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Only embed + add documents if the collection is currently empty, so we
    # don't re-embed (and duplicate) the same PDFs on every run.
    existing_count = vector_store._collection.count()
    if existing_count == 0 and chunks:
        # Chroma caps how many items can be added in a single call, so we
        # add the chunks in smaller batches instead of all at once.
        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            print(f"[INFO] Added batch {i // BATCH_SIZE + 1} "
                  f"({len(batch)} chunks) to Chroma")
        print(f"[INFO] Added {len(chunks)} chunks total to Chroma ({CHROMA_PERSIST_DIR})")
    else:
        print(f"[INFO] Using existing Chroma collection "
              f"({existing_count} chunks already stored)")

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # BM25 needs raw chunks in memory. If this run didn't produce fresh
    # chunks (e.g. Chroma already had data from a previous run), pull the
    # text back out of Chroma so BM25 still has something to index.
    bm25_source_chunks = chunks
    if not bm25_source_chunks:
        existing = vector_store.get()
        texts = existing.get("documents") or []
        bm25_source_chunks = [Document(page_content=t) for t in texts]

    if not bm25_source_chunks:
        print("[WARNING] No chunks available for BM25 — falling back to dense-only retrieval.")
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(bm25_source_chunks)
    bm25_retriever.k = TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[DENSE_WEIGHT, BM25_WEIGHT],
    )
    return ensemble_retriever


retriever = build_retriever()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "information in the context below.\n\n"
    "If the context does not contain relevant information to answer the "
    "question, clearly say that you could not find relevant information in "
    "the documents — do not make up an answer.\n\n"
    "Always respond in exactly this format, nothing else:\n\n"
    "Answer: <your answer to the question, or a not-found message>\n"
    "Source Found: <Yes or No — whether the context actually contained "
    "relevant information>\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

chain = prompt | llm


# Step 3: Single-shot RAG function (retrieve -> generate), traced for DeepEval

@observe(name="rag_agent")
def rag_agent(user_input: str) -> str:
    """Answer ONE question using the RAG pipeline and return the answer."""
    retrieved_docs = retriever.invoke(user_input)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    if not retrieved_chunks:
        print("[WARNING] No documents retrieved for this query.")

    context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant documents found."

    response = chain.invoke({"context": context_text, "question": user_input})
    answer = response.content

    # Push retrieved chunks + final answer into the DeepEval trace so
    # rag_eval.py's metrics (Faithfulness, Contextual Precision/Recall,
    # Answer Relevancy) have retrieval_context and output to grade.
    update_current_trace(
        output=answer,
        retrieval_context=retrieved_chunks if retrieved_chunks else None,
    )

    return answer


if __name__ == "__main__":
    query = input("Ask a question: ")
    print(rag_agent(query))