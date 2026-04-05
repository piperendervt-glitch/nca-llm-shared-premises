"""
外部タスクファイルを読み込む
reference/external_tasks/から読み込み
world_rule + statementを結合してquestionフィールドを作成
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_external_tasks(source: str) -> list[dict]:
    """
    source: "grok" or "chatgpt"
    """
    fname = f"{source}_wc_tasks.jsonl"
    path = REPO_ROOT / "reference" / "external_tasks" / fname

    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line.strip())
            t["question"] = (
                f"World rule: {t['world_rule']}\n"
                f"Statement: {t['statement']}"
            )
            t["task_set"] = f"external_wc_{source}"
            tasks.append(t)
    return tasks
