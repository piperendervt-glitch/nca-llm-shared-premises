"""
run_single_agent_v2.py

MVE-20260404-02: Single-agent baseline (qwen2.5:3b).
Identical to v1 single-agent, output to v2 results dir.
"""

import argparse
import json
import time
from pathlib import Path

import httpx

from task_generator import generate_tasks as generate_world_consistency_tasks

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v2"

SYSTEM_PROMPT = """You are evaluating whether a statement is correct.
Output your response in JSON format:
{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}"""

_client = httpx.Client(timeout=120.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_llm(model: str, task_input: str) -> dict:
    prompt = f"""{SYSTEM_PROMPT}

Task: {task_input}

Respond ONLY in JSON format."""

    response = _client.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def verdict_matches(decision: str, label: bool) -> bool:
    if decision in ("CORRECT", "CONSISTENT"):
        return label is True
    elif decision in ("INCORRECT", "CONTRADICTION"):
        return label is False
    return False


def load_completed(path: Path) -> set:
    completed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add(r["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single-agent baseline for MVE-20260404-02")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g. qwen2.5:3b)")
    args = parser.parse_args()

    model = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "single_qwen3b.jsonl"

    print("=" * 60)
    print(f"Single-agent experiment: {model}")
    print(f"Output: {results_path}")
    print("=" * 60)

    raw_tasks = generate_world_consistency_tasks()
    tasks = []
    for t in raw_tasks:
        tasks.append({
            "task_id": f"wc_{t.task_id:03d}",
            "task_set": "world_consistency",
            "task_type": "world_consistency",
            "question": t.question,
            "world_rule": t.world_rule,
            "label": t.label,
            "task_input": f"World rule: {t.world_rule}\nStatement: {t.question}",
        })

    print(f"Loaded {len(tasks)} tasks")

    completed = load_completed(results_path)
    remaining = [t for t in tasks if t["task_id"] not in completed]

    if not remaining:
        print(f"All {len(tasks)} tasks already completed. Skipping.")
        return

    done = len(completed)
    total = len(tasks)
    correct_count = 0
    print(f"{done}/{total} done, {len(remaining)} remaining.")

    mode = "a" if completed else "w"
    with open(results_path, mode, encoding="utf-8") as f_out:
        for i, task in enumerate(remaining):
            t0 = time.time()
            try:
                output = call_llm(model, task["task_input"])
                decision = output.get("decision", "UNKNOWN")
                confidence = float(output.get("confidence", 0.5))
                reasoning = output.get("reasoning", "")
            except Exception as e:
                decision = "ERROR"
                confidence = 0.0
                reasoning = str(e)
                print(f"  ERROR on {task['task_id']}: {e}")

            elapsed = time.time() - t0
            is_correct = verdict_matches(decision, task["label"])

            record = {
                "task_id": task["task_id"],
                "task_set": task["task_set"],
                "task_type": task["task_type"],
                "question": task["question"],
                "label": task["label"],
                "model": model,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "is_correct": is_correct,
                "elapsed_sec": round(elapsed, 2),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if is_correct:
                correct_count += 1

            if done % 10 == 0 or done == total:
                acc = correct_count / (done - len(completed)) * 100 if (done - len(completed)) > 0 else 0
                print(f"  [{model}] {done}/{total} ({acc:.1f}% acc) last={elapsed:.1f}s")

    print(f"Complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
