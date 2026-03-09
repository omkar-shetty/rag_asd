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

try:
    metric = FaithfulnessMetric(
        threshold=0.7,
        model=GroqJudge(),
        include_reason=True
    )

    test_cases = []

    try:
        with open("data/rag_logs.jsonl") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        test_case = LLMTestCase(
                            input=entry["query"],
                            actual_output=entry["answer"],
                            retrieval_context=entry["retrieved_context"]
                        )
                        test_cases.append(test_case)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON line: {str(e)}")
                        continue
    except FileNotFoundError:
        print("Error: rag_logs.jsonl not found. Please run the app first to generate logs.")
        exit(1)

    if not test_cases:
        print("Error: No valid test cases found in rag_logs.jsonl")
        exit(1)

    print(f"Evaluating {len(test_cases)} test cases...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )

    print(results)
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    exit(1)