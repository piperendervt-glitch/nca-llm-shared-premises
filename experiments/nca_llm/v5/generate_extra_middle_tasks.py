"""Generate 37 extra math_middle tasks (seed=43) for power supplement."""

import json
import random
from middle_school_task_generator import generate_middle_school_tasks

# Generate existing tasks (seed=42) to build exclusion set
existing_tasks = generate_middle_school_tasks(seed=42)
existing_questions = {t.question for t in existing_tasks}

# Generate new pool (seed=43)
new_tasks_pool = generate_middle_school_tasks(seed=43)

# Filter out any that duplicate existing questions
new_tasks = [t for t in new_tasks_pool if t.question not in existing_questions]

# Take first 37
extra_tasks = new_tasks[:37]

# Save as JSONL
with open("extra_middle_tasks.jsonl", "w", encoding="utf-8") as f:
    for i, t in enumerate(extra_tasks):
        record = {
            "task_id": f"mid_extra_{i:03d}",
            "task_set": "math_middle",
            "task_type": t.task_type,
            "question": t.question,
            "world_rule": t.world_rule,
            "label": t.label,
            "level": t.level,
            "difficulty": t.difficulty,
            "answer": t.answer,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Generated {len(extra_tasks)} extra tasks (from pool of {len(new_tasks)} non-duplicate)")
print(f"Task IDs: {[f'mid_extra_{i:03d}' for i in range(min(3, len(extra_tasks)))]}")

correct = sum(1 for t in extra_tasks if t.label)
print(f"Labels: {correct} CORRECT, {len(extra_tasks) - correct} INCORRECT")
