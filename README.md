# RAG-Question-Answering-and-Evaluation-System
A hybrid Retrieval-Augmented Generation (RAG) pipeline that combines dense semantic search with BM25 keyword retrieval to answer questions from a set of PDF documents, paired with a white-box DeepEval evaluation script that grades retrieval quality and answer faithfulness.

Overview

This project demonstrates a standalone RAG workflow with built-in evaluation:

Loads and chunks PDF documents from a local folder
Embeds chunks with OpenAI embeddings and stores them in a persistent ChromaDB vector store
Retrieves relevant chunks using an EnsembleRetriever (70% dense / 30% BM25)
Generates grounded answers with ChatOpenAI, explicitly refusing to answer when no relevant context is found
Traces every run with DeepEval (@observe + update_current_trace) so retrieval and generation can both be evaluated
Evaluates responses using Contextual Precision, Answer Relevancy, and Faithfulness metrics

Metrics
Contextual Precision — are the most relevant retrieved chunks ranked highest
Answer Relevancy — does the answer actually address the question
Faithfulness — is the answer grounded in the retrieved context (catches hallucination)
