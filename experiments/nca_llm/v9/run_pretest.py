"""
run_pretest.py

Pre-test: single qwen2.5:7b agent on 100 candidate causal tasks.
temperature=0 (deterministic) to measure baseline difficulty.
Selects tasks where overall accuracy is 70-85%.

Output:
  results/nca_llm/v9/pretest_results.jsonl (all 100)
  results/nca_llm/v9/selected_tasks.jsonl (filtered)
"""

import json
import time
from pathlib import Path

import httpx

from causal_task_generator import generate_tasks

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v9"

_client = httpx.Client(timeout=120.0)


def call_llm(prompt: str) -> dict:
    response = _client.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
              "options": {"temperature": 0}},
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def make_prompt(task_input: str) -> str:
    return f"""You are evaluating a causal reasoning statement.
{task_input}

Based on common sense and causal reasoning, determine if this statement is CORRECT or INCORRECT.
Think carefully about causation vs correlation, necessity vs sufficiency, and direct vs indirect causes.

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation in 2-3 sentences"
}}"""


def verdict_matches(verdict: str, label: bool) -> bool:
    if verdict in ("CORRECT", "CONSISTENT"):
        return label is True
    elif verdict in ("INCORRECT", "CONTRADICTION"):
        return label is False
    return False


def main():
    tasks = generate_tasks()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pretest_path = RESULTS_DIR / "pretest_results.jsonl"

    print("=" * 60)
    print(f"MVE-09 Pre-test: {MODEL} (temperature=0)")
    print(f"Tasks: {len(tasks)} candidates")
    print(f"Output: {pretest_path}")
    print("=" * 60)

    # Check for existing results
    completed = set()
    if pretest_path.exists():
        with open(pretest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed.add(json.loads(line.strip())["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    remaining = [t for t in tasks if t.task_id not in completed]
    if not remaining:
        print(f"All {len(tasks)} tasks completed.")
    else:
        done = len(completed)
        print(f"{done}/{len(tasks)} done, {len(remaining)} remaining.")

        mode = "a" if completed else "w"
        with open(pretest_path, mode, encoding="utf-8") as f_out:
            for task in remaining:
                t0 = time.time()
                result = call_llm(make_prompt(task.question))
                elapsed = time.time() - t0

                decision = result.get("decision", "UNKNOWN")
                is_correct = verdict_matches(decision, task.label)

                record = {
                    "task_id": task.task_id,
                    "task_set": task.task_set,
                    "task_type": task.task_type,
                    "question": task.question,
                    "label": task.label,
                    "prediction": decision,
                    "is_correct": is_correct,
                    "confidence": float(result.get("confidence", 0.5)),
                    "elapsed_sec": round(elapsed, 2),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] last={elapsed:.1f}s")

    # Analyze and select
    results = []
    with open(pretest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))

    n_total = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / n_total * 100 if n_total else 0

    print(f"\n{'='*60}")
    print(f"Pre-test Results:")
    print(f"  Total: {n_total} tasks")
    print(f"  Correct: {n_correct}")
    print(f"  Accuracy: {accuracy:.1f}%")

    # Per-type breakdown
    types = {}
    for r in results:
        tt = r["task_type"]
        if tt not in types:
            types[tt] = {"total": 0, "correct": 0}
        types[tt]["total"] += 1
        if r["is_correct"]:
            types[tt]["correct"] += 1

    print(f"\n  Per-type accuracy:")
    for tt in sorted(types.keys()):
        acc = types[tt]["correct"] / types[tt]["total"] * 100
        print(f"    {tt}: {types[tt]['correct']}/{types[tt]['total']} ({acc:.1f}%)")

    # Selection: use ALL tasks if overall accuracy is 70-85%
    # Otherwise filter by type to get into range
    if 70 <= accuracy <= 85:
        selected = results
        print(f"\n  Overall accuracy {accuracy:.1f}% is in range [70-85%].")
        print(f"  Using all {len(selected)} tasks.")
    else:
        # Use all tasks anyway but report the issue
        selected = results
        print(f"\n  WARNING: Overall accuracy {accuracy:.1f}% is outside [70-85%].")
        print(f"  Using all {len(selected)} tasks (reporting to design chat).")

    # Save selected tasks
    selected_path = RESULTS_DIR / "selected_tasks.jsonl"
    with open(selected_path, "w", encoding="utf-8") as f:
        for r in selected:
            task_record = {
                "task_id": r["task_id"],
                "task_set": r["task_set"],
                "task_type": r["task_type"],
                "question": r["question"],
                "label": r["label"],
            }
            f.write(json.dumps(task_record, ensure_ascii=False) + "\n")

    print(f"\n  Selected tasks saved to: {selected_path}")
    print(f"  Selected count: {len(selected)} (threshold: >=50)")
    print(f"  Proceed to main experiment: {'YES' if len(selected) >= 50 else 'NO (need redesign)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
