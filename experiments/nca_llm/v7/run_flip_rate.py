"""
run_flip_rate.py

MVE-20260404-07: Measure flip rate (single model instability)
for each task across 5 independent trials with temperature=0.8.

Usage:
  python run_flip_rate.py --task world_consistency
  python run_flip_rate.py --task logic_syllogism
  python run_flip_rate.py --task logic_contradiction
  python run_flip_rate.py --task math_middle
"""

import argparse
import json
import time
from pathlib import Path

import httpx

MODEL = "qwen2.5:7b"
N_TRIALS = 5
TEMPERATURE = 0.8
OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v7"

_client = httpx.Client(timeout=120.0)


def call_ollama(prompt: str) -> dict:
    response = _client.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": TEMPERATURE},
        },
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def make_prompt(task_input: str) -> str:
    return f"""You are evaluating a logical statement.
Task: {task_input}

Determine if the statement is CORRECT or INCORRECT based on the given rule.

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation in 1-2 sentences"
}}"""


def load_tasks(task_name: str) -> list:
    if task_name == "world_consistency":
        from task_generator import generate_tasks
        raw = generate_tasks()
        return [{
            "task_id": f"wc_{t.task_id:03d}" if isinstance(t.task_id, int) else str(t.task_id),
            "task_set": "world_consistency",
            "task_type": "world_consistency",
            "question": t.question,
            "label": t.label,
            "task_input": f"World rule: {t.world_rule}\nStatement: {t.question}",
        } for t in raw]

    elif task_name in ("logic_syllogism", "logic_contradiction"):
        target_type = task_name.replace("logic_", "")
        logic_path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v6" / "logic_7b_homo.jsonl"
        tasks = []
        seen = set()
        with open(logic_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                if r["task_type"] == target_type and r["task_id"] not in seen:
                    seen.add(r["task_id"])
                    tasks.append({
                        "task_id": r["task_id"],
                        "task_set": f"logic_{target_type}",
                        "task_type": target_type,
                        "question": r["question"],
                        "label": r["label"],
                        "task_input": r["question"],
                    })
        return tasks

    elif task_name == "math_middle":
        from middle_school_task_generator import generate_middle_school_tasks
        raw = generate_middle_school_tasks()
        return [{
            "task_id": f"mid_{t.task_id:03d}" if isinstance(t.task_id, int) else str(t.task_id),
            "task_set": "math_middle",
            "task_type": getattr(t, "task_type", "math_middle"),
            "question": t.question,
            "label": t.label,
            "task_input": f"Rule: {t.world_rule}\nStatement: {t.question}",
        } for t in raw]

    else:
        raise ValueError(f"Unknown task: {task_name}")


OUTPUT_FILES = {
    "world_consistency": "flip_wc.jsonl",
    "logic_syllogism": "flip_syllogism.jsonl",
    "logic_contradiction": "flip_contradiction.jsonl",
    "math_middle": "flip_math_mid.jsonl",
}


def load_completed(path: Path) -> set:
    completed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["world_consistency", "logic_syllogism",
                                 "logic_contradiction", "math_middle"])
    args = parser.parse_args()

    tasks = load_tasks(args.task)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / OUTPUT_FILES[args.task]

    print("=" * 60)
    print(f"MVE-07 flip rate: {args.task}")
    print(f"Model: {MODEL}, Trials: {N_TRIALS}, Temperature: {TEMPERATURE}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {results_path}")
    print("=" * 60)

    completed = load_completed(results_path)
    remaining = [t for t in tasks if t["task_id"] not in completed]
    if not remaining:
        print(f"All {len(tasks)} tasks completed. Skipping.")
        return

    done = len(completed)
    total = len(tasks)
    print(f"{done}/{total} done, {len(remaining)} remaining.")

    mode = "a" if completed else "w"
    with open(results_path, mode, encoding="utf-8") as f_out:
        for task in remaining:
            t0 = time.time()
            trials = []
            for trial_n in range(1, N_TRIALS + 1):
                try:
                    result = call_ollama(make_prompt(task["task_input"]))
                except Exception as e:
                    result = {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": str(e)}
                trials.append({
                    "trial": trial_n,
                    "decision": result.get("decision", "UNKNOWN"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                })

            decisions = [t["decision"] for t in trials]
            n_correct = decisions.count("CORRECT")
            n_incorrect = decisions.count("INCORRECT")
            n_valid = n_correct + n_incorrect
            p_correct = n_correct / n_valid if n_valid else 0.5
            flip_rate = 1 - max(p_correct, 1 - p_correct)

            # Majority vote correctness
            majority = "CORRECT" if n_correct >= n_incorrect else "INCORRECT"
            if majority in ("CORRECT", "CONSISTENT"):
                is_correct_majority = task["label"] is True
            else:
                is_correct_majority = task["label"] is False

            elapsed = time.time() - t0
            record = {
                "task_id": task["task_id"],
                "task_set": task["task_set"],
                "task_type": task["task_type"],
                "question": task["question"],
                "label": task["label"],
                "trials": trials,
                "p_correct": round(p_correct, 2),
                "flip_rate": round(flip_rate, 2),
                "is_correct_majority": is_correct_majority,
                "elapsed_sec": round(elapsed, 2),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{args.task}] {done}/{total} flip_rate={flip_rate:.2f} last={elapsed:.1f}s")

    print(f"Complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
