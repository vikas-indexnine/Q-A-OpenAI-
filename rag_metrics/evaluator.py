import os
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
AnswerRelevancyMetric,
FaithfulnessMetric,
ContextualRecallMetric,
ContextualPrecisionMetric,
ContextualRelevancyMetric
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

async def evaluate_rag_response(question, context, model_output, expected_output):
	test_case = LLMTestCase(
		input=question,
		actual_output=model_output,
		expected_output=expected_output,
		retrieval_context=context,
	)