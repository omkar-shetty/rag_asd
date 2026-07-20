import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from ..logger_config import setup_script_logger

load_dotenv()
logger = setup_script_logger("evaluate_rag")


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model="llama-3.3-70b-versatile"):
        self.model_name = model
        self.client = None
        self.load_model()

    def load_model(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
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


def extract_verdict_counts(verbose_logs: str) -> dict | None:
    """Extract yes/idk/no verdict counts from DeepEval verbose_logs."""
    if not verbose_logs or "Verdicts:" not in verbose_logs:
        return None
    verdict_section = verbose_logs[verbose_logs.index("Verdicts:"):]
    return {
        "yes": verdict_section.count('"verdict": "yes"'),
        "idk": verdict_section.count('"verdict": "idk"'),
        "no":  verdict_section.count('"verdict": "no"'),
    }


def main():
    try:
        metric = FaithfulnessMetric(
            threshold=0.7,
            model=GroqJudge(),
            include_reason=True
        )
    except Exception as ex:
        logger.error(f"Failed to initialise evaluation metric: {ex}")
        sys.exit(1)

    test_cases = []
    try:
        with open("data/rag_logs_eval.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    test_cases.append(LLMTestCase(
                        input=entry["query"],
                        actual_output=entry["answer"],
                        retrieval_context=entry["retrieved_context"]
                    ))
                except (json.JSONDecodeError, KeyError) as ex:
                    logger.warning(f"Skipping malformed log line: {ex}")
    except FileNotFoundError:
        logger.error("data/rag_logs.jsonl not found. Run the app first to generate logs.")
        sys.exit(1)

    if not test_cases:
        logger.error("No valid test cases found in rag_logs.jsonl")
        sys.exit(1)

    print(f"\nEvaluating {len(test_cases)} test cases...")
    try:
        results = evaluate(test_cases=test_cases, metrics=[metric])
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

    #Parse results
    test_results = results.test_results
    eval_model = metric.model.get_model_name()

    print(f"Faithfulness evaluation complete ({eval_model} judge)\n")

    log_entries = []
    passed = 0
    perfect = 0
    all_verdicts = {"yes": 0, "idk": 0, "no": 0}
    total_claims = 0

    for tr in test_results:
        md = tr.metrics_data[0] if tr.metrics_data else None
        score     = md.score     if md else None
        threshold = md.threshold if md else None
        success   = md.success   if md else None
        reason    = md.reason    if md else None

        verdicts = None
        if md and md.verbose_logs:
            verdicts = extract_verdict_counts(md.verbose_logs)
            if verdicts is None:
                logger.warning(f"Verdict stats not available for: '{tr.input[:60]}'")
        else:
            logger.warning(f"Verdict stats not available for: '{tr.input[:60]}'")

        if verdicts:
            for k in all_verdicts:
                all_verdicts[k] += verdicts[k]
            total_claims += sum(verdicts.values())

        if success:
            passed += 1
        if score == 1.0:
            perfect += 1

        log_entries.append({
            "input":            tr.input,
            "actual_output":    tr.actual_output,
            "score":            score,
            "threshold":        threshold,
            "success":          success,
            "reason":           reason,
            "evaluation_model": eval_model,
            "verdicts":         verdicts,
        })

    # Write to JSON log
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    log_path = f"logs/evaluate_rag_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)

    #Console summary
    n = len(test_results)
    pass_pct = round(100 * passed / n) if n else 0
    perfect_count = perfect
    mean_claims = round(total_claims / n, 1) if n else 0

    total_v = sum(all_verdicts.values())
    if total_v:
        yes_pct = round(100 * all_verdicts["yes"] / total_v)
        idk_pct = round(100 * all_verdicts["idk"] / total_v)
        no_pct  = round(100 * all_verdicts["no"]  / total_v)
        verdict_line = f"{yes_pct}% yes, {idk_pct}% idk, {no_pct}% no"
    else:
        verdict_line = "n/a"

    threshold_val = metric.threshold

    print("=== Summary ===")
    print(f"Pass rate: {pass_pct}% ({passed}/{n} passed threshold {threshold_val})")
    print(f"Perfect scores (1.0): {perfect_count}/{n}")
    print(f"Mean claims per case: {mean_claims}")
    print(f"Verdicts across all cases: {verdict_line}")
    print(f"\nFull results: {log_path}")


if __name__ == "__main__":
    main()
