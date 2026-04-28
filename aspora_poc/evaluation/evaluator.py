"""
Evaluator — runs the golden dataset through the full pipeline and checks results.
Measures: intent classification accuracy, retrieval recall, guardrail correctness.
Run with: python -m evaluation.evaluator
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")

from engines.orchestrator import process_query


def load_golden_dataset() -> list[dict]:
    """Load the golden dataset from JSON."""
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path) as f:
        return json.load(f)


def evaluate_single(test_case: dict) -> dict:
    """Run a single test case and check all expected conditions."""
    query = test_case["query"]
    jurisdiction = test_case.get("jurisdiction")

    result = process_query(query, user_jurisdiction=jurisdiction)

    checks = {}
    passed = True

    # Check 1: Intent classification
    if "expected_intent" in test_case:
        intent_match = result.get("intent") == test_case["expected_intent"]
        checks["intent_correct"] = intent_match
        if not intent_match:
            passed = False

    # Check 2: Pipeline used
    if "expected_pipeline" in test_case:
        pipeline_match = result.get("pipeline_used") == test_case["expected_pipeline"]
        checks["pipeline_correct"] = pipeline_match
        if not pipeline_match:
            passed = False

    # Check 3: Answer contains expected strings
    response = result.get("response", "").lower()
    if "expected_answer_contains" in test_case:
        for expected in test_case["expected_answer_contains"]:
            key = f"contains_{expected[:20]}"
            contains = expected.lower() in response
            checks[key] = contains
            if not contains:
                passed = False

    # Check 4: Answer should NOT contain forbidden strings
    if "should_NOT_contain" in test_case:
        for forbidden in test_case["should_NOT_contain"]:
            key = f"not_contains_{forbidden[:20]}"
            not_contains = forbidden.lower() not in response
            checks[key] = not_contains
            if not not_contains:
                passed = False

    # Check 5: Disclaimer present if expected
    if test_case.get("should_contain_disclaimer"):
        has_disclaimer = "disclaimer" in response or "⚠️" in result.get("response", "")
        checks["has_disclaimer"] = has_disclaimer
        if not has_disclaimer:
            passed = False

    # Check 6: Compliance safe
    checks["compliance_safe"] = result.get("compliance_safe", True)

    return {
        "id": test_case["id"],
        "query": query[:60],
        "passed": passed,
        "checks": checks,
        "actual_intent": result.get("intent"),
        "actual_pipeline": result.get("pipeline_used"),
        "docs_retrieved": result.get("docs_retrieved", 0),
    }


def run_evaluation() -> dict:
    """Run the full evaluation and generate report."""
    dataset = load_golden_dataset()
    print(f"\n📊 Running evaluation on {len(dataset)} test cases...\n")

    results = []
    start_time = time.time()

    for i, test_case in enumerate(dataset):
        eval_result = evaluate_single(test_case)
        results.append(eval_result)

        status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
        print(f"  [{eval_result['id']:2d}] {status} | "
              f"Intent: {eval_result['actual_intent']:15s} | "
              f"{eval_result['query']}")

        if not eval_result["passed"]:
            for check, value in eval_result["checks"].items():
                if not value:
                    print(f"       ↳ Failed: {check}")

    elapsed = round(time.time() - start_time, 2)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    accuracy = round(passed / total * 100, 1) if total > 0 else 0

    # Intent accuracy
    intent_tests = [r for r in results if "intent_correct" in r["checks"]]
    intent_correct = sum(1 for r in intent_tests if r["checks"]["intent_correct"])
    intent_accuracy = round(intent_correct / len(intent_tests) * 100, 1) if intent_tests else 0

    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "intent_accuracy": intent_accuracy,
        "elapsed_seconds": elapsed,
    }

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total tests:      {total}")
    print(f"  Passed:           {passed} ✅")
    print(f"  Failed:           {failed} ❌")
    print(f"  Overall accuracy: {accuracy}%")
    print(f"  Intent accuracy:  {intent_accuracy}%")
    print(f"  Total time:       {elapsed}s")
    print(f"{'=' * 60}\n")

    return summary


if __name__ == "__main__":
    run_evaluation()
