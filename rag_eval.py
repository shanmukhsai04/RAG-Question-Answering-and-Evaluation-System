"""
rag_eval.py
===========
White box evaluation of the RAG pipeline in rag.py using DeepEval.

Metrics used:
  - ContextualPrecisionMetric : are the most relevant retrieved chunks
                                 ranked highest (grades retrieval quality)
  - AnswerRelevancyMetric     : does the final answer actually address
                                 the question
  - FaithfulnessMetric        : is the answer grounded in the retrieved
                                 context (catches hallucination)

How it works:
  rag_eval_wrapper() is @observe-traced and, for each golden, attaches the
  golden's expected_output to the current trace. rag_agent() (imported from
  rag.py) is also @observe-traced and pushes retrieval_context + output
  into the same trace. DeepEval's evals_iterator() then builds a test case
  per golden from the trace and grades it against all three metrics.
"""

from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.tracing import observe, update_current_trace
 
from rag import rag_agent as _rag_agent
 
 
# attaches expected_output to the trace so the metrics can grade against it
@observe(name="rag_eval_wrapper")
def rag_eval_wrapper(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_output:
        update_current_trace(expected_output=golden.expected_output)
    return _rag_agent(user_input)
 
 
# test questions + expected answers (more in rag_eval_test_cases.md)
dataset = EvaluationDataset(goldens=[
    # Question 1
    Golden(
        input="Question 1",
        expected_output="<fill in from the PDF>",
    ),
    # Question 2
    Golden(
        input="Question 2",
        expected_output="<fill in from the PDF>",
    ),
    # Question 3
    Golden(
        input="Question 3",
        expected_output="<fill in from the PDF>",
    ),
    # Question 4
    Golden(
        input="Question 4",
        expected_output="<fill in from the PDF>",
    ),
    # Question 5
    Golden(
        input="Question 5",
        expected_output="<fill in from the PDF>",
    ),
])
 
precisionMetric = ContextualPrecisionMetric(threshold=0.7, model="gpt-4o", include_reason=True)
relevancyMetric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o", include_reason=True)
faithfulMetric = FaithfulnessMetric(threshold=0.7, model="gpt-4o", include_reason=True)
 
for golden in dataset.evals_iterator(
    metrics=[precisionMetric, relevancyMetric, faithfulMetric]
):
    rag_eval_wrapper(golden.input)