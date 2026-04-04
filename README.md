# nca-llm-shared-premises

Verification laboratory for North Star A-4:
"Most AI failures result from humans not updating
the contract (shared_premises) with AI."

All experiments are designed with MVE sheets before execution.
Human approval required before running any experiment.
Only ADVANCE-judged experiments are run.

## North Star

A-4: "Most AI failures result from humans not updating
the contract (shared_premises) with AI."

## Relationship to Previous Repository

- **nca-llm-experiment**: Exploration phase (v1-v11)
  Free-form experiments, pattern discovery
- **nca-llm-shared-premises**: Verification phase
  MVE-sheet-driven, north-star-connected experiments only

## Structure

- `docs/` — MVE sheets and design documents
- `experiments/` — Experiment code
- `results/` — Raw results (jsonl)
- `reference/` — Exploration-phase reference data
- `scripts/` — verify_results.py and utilities
