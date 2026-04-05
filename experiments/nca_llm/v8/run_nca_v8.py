"""
run_nca_v8.py

MVE-20260404-08: Common sense reasoning diversity effect.
No World rule - pure common sense judgment.

Usage:
  python run_nca_v8.py --condition 7b_homo
  python run_nca_v8.py --condition 7b_het
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from common_sense_task_generator import generate_tasks

OLLAMA_URL = "http://localhost:11434/api/generate"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v8"

AGREE = [30, 80, 80]
STEPS = 3

CONDITIONS = {
    "7b_homo": ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"],
    "7b_het":  ["qwen2.5:7b", "llama3:latest", "mistral:7b"],
}

_client = httpx.Client(timeout=120.0)


def call_llm(model: str, prompt: str) -> dict:
    response = _client.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def format_output(output: dict) -> str:
    return (f"{output.get('decision', 'UNKNOWN')} "
            f"(confidence: {output.get('confidence', 0.5)}) - "
            f"{output.get('reasoning', '')}")


def agreement_instruction(pct: int) -> str:
    if pct >= 50:
        return (f"You should agree with your teammates approximately {pct}% of the time.\n"
                f"If they strongly agree, consider following their consensus.")
    return (f"You should agree with your teammates approximately {pct}% of the time.\n"
            f"If they strongly agree, consider maintaining your independent judgment.")


def solver_prompt(task_input: str, agree_pct: int) -> str:
    agree = agreement_instruction(agree_pct)
    return f"""You are the Solver in a 3-node reasoning network.
Task: {task_input}

Based on common sense and general knowledge, determine if this statement is CORRECT or INCORRECT.
Think step by step.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation in 2-3 sentences"
}}"""


def verifier_prompt(task_input: str, solver_out: dict, agree_pct: int) -> str:
    agree = agreement_instruction(agree_pct)
    return f"""You are the Verifier in a 3-node reasoning network.
Task: {task_input}

The Solver's answer: {format_output(solver_out)}

Independently verify this using your own common sense and knowledge.
Do NOT simply agree - think for yourself.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your independent verification in 2-3 sentences"
}}"""


def critic_prompt(task_input: str, solver_out: dict, verifier_out: dict, agree_pct: int) -> str:
    agree = agreement_instruction(agree_pct)
    return f"""You are the Critic in a 3-node reasoning network.
Task: {task_input}

Solver's reasoning: {format_output(solver_out)}
Verifier's reasoning: {format_output(verifier_out)}

Critically evaluate both answers using common sense.
If they disagree, determine who is right.
If they agree but seem wrong, say so.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Final judgment with brief explanation"
}}"""


def run_nca(task_input: str, models: list[str]) -> dict:
    all_steps = []
    for step in range(STEPS):
        s_out = call_llm(models[0], solver_prompt(task_input, AGREE[0]))
        v_out = call_llm(models[1], verifier_prompt(task_input, s_out, AGREE[1]))
        c_out = call_llm(models[2], critic_prompt(task_input, s_out, v_out, AGREE[2]))
        all_steps.append({
            "step": step,
            "solver": {"model": models[0], "output": s_out},
            "verifier": {"model": models[1], "output": v_out},
            "critic": {"model": models[2], "output": c_out},
        })

    last = all_steps[-1]
    node_outputs = {
        "solver": {"decision": last["solver"]["output"].get("decision", "UNKNOWN"),
                    "confidence": float(last["solver"]["output"].get("confidence", 0.5))},
        "verifier": {"decision": last["verifier"]["output"].get("decision", "UNKNOWN"),
                      "confidence": float(last["verifier"]["output"].get("confidence", 0.5))},
        "critic": {"decision": last["critic"]["output"].get("decision", "UNKNOWN"),
                    "confidence": float(last["critic"]["output"].get("confidence", 0.5))},
    }
    votes = [node_outputs[r]["decision"] for r in ("solver", "verifier", "critic")]
    vote_dist = {"CORRECT": votes.count("CORRECT"), "INCORRECT": votes.count("INCORRECT")}
    filtered = [v for v in votes if v in ("CORRECT", "INCORRECT")]
    verdict = Counter(filtered).most_common(1)[0][0] if filtered else "UNKNOWN"
    is_unanimous = any(v == 3 for v in vote_dist.values())

    return {"verdict": verdict, "vote_distribution": vote_dist,
            "is_unanimous": is_unanimous, "node_outputs": node_outputs}


def verdict_matches(verdict: str, label: bool) -> bool:
    if verdict in ("CORRECT", "CONSISTENT"):
        return label is True
    elif verdict in ("INCORRECT", "CONTRADICTION"):
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
                    completed.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def main():
    parser = argparse.ArgumentParser(description="MVE-08 common sense NCA experiment")
    parser.add_argument("--condition", required=True, choices=["7b_homo", "7b_het"])
    args = parser.parse_args()

    models = CONDITIONS[args.condition]
    output_file = f"common_sense_{args.condition}.jsonl"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / output_file

    raw_tasks = generate_tasks()
    tasks = [{
        "task_id": t.task_id,
        "task_set": t.task_set,
        "task_type": t.task_type,
        "question": t.question,
        "label": t.label,
        "task_input": t.question,
    } for t in raw_tasks]

    print("=" * 60)
    print(f"MVE-08: common_sense x {args.condition}")
    print(f"Models: {models}")
    print(f"Steps: {STEPS}, Agree: {AGREE}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {results_path}")
    print("=" * 60)

    completed = load_completed(results_path)
    remaining = [t for t in tasks if t["task_id"] not in completed]
    if not remaining:
        print(f"All {len(tasks)} tasks completed. Skipping.")
        return

    done, total, correct_count = len(completed), len(tasks), 0
    print(f"{done}/{total} done, {len(remaining)} remaining.")

    tag = f"cs-{args.condition}"
    mode = "a" if completed else "w"
    with open(results_path, mode, encoding="utf-8") as f_out:
        for task in remaining:
            t0 = time.time()
            try:
                result = run_nca(task["task_input"], models)
            except Exception as e:
                result = {"verdict": "ERROR", "vote_distribution": {"CORRECT": 0, "INCORRECT": 0},
                          "is_unanimous": False, "node_outputs": {}}
                print(f"  ERROR on {task['task_id']}: {e}")

            elapsed = time.time() - t0
            is_correct = verdict_matches(result["verdict"], task["label"])
            record = {
                "task_id": task["task_id"], "task_set": task["task_set"],
                "task_type": task["task_type"], "question": task["question"],
                "label": task["label"], "prediction": result["verdict"],
                "is_correct": is_correct, "vote_distribution": result["vote_distribution"],
                "is_unanimous": result["is_unanimous"], "node_outputs": result["node_outputs"],
                "condition": args.condition,
                "models_used": {"solver": models[0], "verifier": models[1], "critic": models[2]},
                "elapsed_sec": round(elapsed, 2),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if is_correct:
                correct_count += 1
            if done % 10 == 0 or done == total:
                acc = correct_count / (done - len(completed)) * 100 if (done - len(completed)) > 0 else 0
                print(f"  [{tag}] {done}/{total} ({acc:.1f}% acc) last={elapsed:.1f}s")

    print(f"Complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
