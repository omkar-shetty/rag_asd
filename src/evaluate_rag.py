import os
import json
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model="llama-3.1-8b-instant"):
        self.model_name = model
        self.client = None
        self.load_model()

    def load_model(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self.client

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

metric = FaithfulnessMetric(
    threshold=0.7,
    model=GroqJudge(),
    include_reason=True
)

test_cases = []

with open("data/rag_logs.jsonl") as f:
    for line in f:
        entry = json.loads(line)

        test_case = LLMTestCase(
            input=entry["query"],
            actual_output=entry["answer"],
            retrieval_context=entry["retrieved_context"]
        )

        test_cases.append(test_case)

results = evaluate(
    test_cases=test_cases,
    metrics=[metric]
)

print(results)