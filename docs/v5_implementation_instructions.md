# 実装指示書: MVE-20260404-05
# NCA-LLM v5: 多様性効果のタスク種別依存性の確認

## 概要

MVE-20260404-05（ADVANCE済み）の実験を実装する。

北極星: A-4
Claim: 「7b異種NCAの多様性効果（CFR低下）は
        world_consistencyだけでなく
        数学タスクでも観察される」

---

## 実験の全体像

### 流用データ（新規実験不要）

```
world_consistency × 4条件（v1〜v4済み）:
  3b同種:   results/nca_llm/v2/nca_3b_results.jsonl
  3b真異種: results/nca_llm/v4/nca_3b_true_het_results.jsonl
  7b同種:   results/nca_llm/v2/nca_7b_results.jsonl
  7b異種:   results/nca_llm/v1/nca_v1_results.jsonl
```

### 新規実験（12実験）

```
優先度1（最重要・先に実行）:
  7b同種 × math_elementary   → results/nca_llm/v5/math_elem_7b_homo.jsonl
  7b同種 × math_middle       → results/nca_llm/v5/math_mid_7b_homo.jsonl
  7b同種 × math_high         → results/nca_llm/v5/math_high_7b_homo.jsonl
  7b異種 × math_elementary   → results/nca_llm/v5/math_elem_7b_het.jsonl
  7b異種 × math_middle       → results/nca_llm/v5/math_mid_7b_het.jsonl
  7b異種 × math_high         → results/nca_llm/v5/math_high_7b_het.jsonl

優先度2（補完・後で実行）:
  3b同種 × math_elementary   → results/nca_llm/v5/math_elem_3b_homo.jsonl
  3b同種 × math_middle       → results/nca_llm/v5/math_mid_3b_homo.jsonl
  3b同種 × math_high         → results/nca_llm/v5/math_high_3b_homo.jsonl
  3b真異種 × math_elementary → results/nca_llm/v5/math_elem_3b_het.jsonl
  3b真異種 × math_middle     → results/nca_llm/v5/math_mid_3b_het.jsonl
  3b真異種 × math_high       → results/nca_llm/v5/math_high_3b_het.jsonl
```

---

## モデル構成

```
3b同種:
  qwen2.5:3b × 3（Solver/Verifier/Critic）

3b真異種:
  qwen2.5:3b（Solver）
  llama3.2:3b（Verifier）
  granite3.1-moe:3b（Critic）

7b同種:
  qwen2.5:7b × 3（Solver/Verifier/Critic）

7b異種:
  qwen2.5:7b（Solver）
  llama3:latest（Verifier）
  mistral:7b（Critic）
```

---

## ファイル構成

```
nca-llm-shared-premises/
├── experiments/nca_llm/v5/
│   ├── math_task_generator.py         # elementary（コピー済み）
│   ├── middle_school_task_generator.py # middle（コピー済み）
│   ├── high_school_task_generator.py   # high（コピー済み）
│   ├── run_nca_v5.py                  # 全条件を一括実行
│   └── analyze_v5.py                  # 全16条件の分析
└── results/nca_llm/v5/
    ├── math_elem_7b_homo.jsonl
    ├── math_elem_7b_het.jsonl
    ├── math_mid_7b_homo.jsonl
    ├── math_mid_7b_het.jsonl
    ├── math_high_7b_homo.jsonl
    ├── math_high_7b_het.jsonl
    ├── math_elem_3b_homo.jsonl
    ├── math_elem_3b_het.jsonl
    ├── math_mid_3b_homo.jsonl
    ├── math_mid_3b_het.jsonl
    ├── math_high_3b_homo.jsonl
    └── math_high_3b_het.jsonl
```

---

## 実装仕様

### run_nca_v5.py

```python
# 設定をパラメータで切り替えられる汎用スクリプト

# 使い方:
# python run_nca_v5.py --task math_elementary --condition 7b_homo
# python run_nca_v5.py --task math_elementary --condition 7b_het
# python run_nca_v5.py --task math_middle --condition 7b_homo
# ... etc

# 条件の定義:
CONDITIONS = {
    "7b_homo": {
        "models": ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"],
        "roles": ["Solver", "Verifier", "Critic"]
    },
    "7b_het": {
        "models": ["qwen2.5:7b", "llama3:latest", "mistral:7b"],
        "roles": ["Solver", "Verifier", "Critic"]
    },
    "3b_homo": {
        "models": ["qwen2.5:3b", "qwen2.5:3b", "qwen2.5:3b"],
        "roles": ["Solver", "Verifier", "Critic"]
    },
    "3b_het": {
        "models": ["qwen2.5:3b", "llama3.2:3b", "granite3.1-moe:3b"],
        "roles": ["Solver", "Verifier", "Critic"]
    }
}

# タスクの定義:
TASKS = {
    "math_elementary": {
        "generator": "math_task_generator",
        "function": "generate_tasks",
        "output_prefix": "math_elem"
    },
    "math_middle": {
        "generator": "middle_school_task_generator",
        "function": "generate_tasks",
        "output_prefix": "math_mid"
    },
    "math_high": {
        "generator": "high_school_task_generator",
        "function": "generate_tasks",
        "output_prefix": "math_high"
    }
}

# 共通設定（v1〜v4と同一）:
steps = 3
agree = [30, 80, 80]
aggregation = "simple_majority"  # Counter

# 出力ファイル名:
# results/nca_llm/v5/{output_prefix}_{condition}.jsonl
# 例: results/nca_llm/v5/math_elem_7b_homo.jsonl

# レコード構造（v1〜v4と同一）:
{
    "task_id": str,
    "task_set": str,      # "math_elementary" etc
    "task_type": str,
    "question": str,
    "label": bool,
    "prediction": str,
    "is_correct": bool,
    "vote_distribution": {"CORRECT": int, "INCORRECT": int},
    "is_unanimous": bool,
    "node_outputs": {
        "solver":   {"decision": str, "confidence": float},
        "verifier": {"decision": str, "confidence": float},
        "critic":   {"decision": str, "confidence": float}
    },
    "condition": str,     # "7b_homo" etc
    "models_used": {
        "solver": str,
        "verifier": str,
        "critic": str
    },
    "elapsed_sec": float
}
```

### analyze_v5.py

```python
# 全16条件（WC×4 + 数学×12）を分析

# 出力フォーマット:
"""
================================================================
MVE-20260404-05 Results: Diversity Effect by Task Type
================================================================

CFR by condition and task set:

Task Set        | 3b同種  | 3b真異種 | 7b同種  | 7b異種  | 7b差分
----------------|---------|---------|---------|---------|-------
world_consist   | 38.5%  | 38.6%  | 30.6%  | 14.3%  | -16.3pp ✅
math_elementary | ??.?%  | ??.?%  | ??.?%  | ??.?%  | ?.?pp  ?
math_middle     | ??.?%  | ??.?%  | ??.?%  | ??.?%  | ?.?pp  ?
math_high       | ??.?%  | ??.?%  | ??.?%  | ??.?%  | ?.?pp  ?

================================================================
必須指標: 7b異種 CFR < 7b同種 CFR（方向が一致）

  math_elementary: 7b異種 ??.?% vs 7b同種 ??.?% → ✅/❌
  math_middle:     7b異種 ??.?% vs 7b同種 ??.?% → ✅/❌
  math_high:       7b異種 ??.?% vs 7b同種 ??.?% → ✅/❌

  3タスク全て一致 → FULL SUCCESS
  2タスク一致    → PARTIAL
  1タスク以下    → FAIL

================================================================
参考: p値（各タスク・7b同種 vs 7b異種）
  math_elementary: p=0.??
  math_middle:     p=0.??
  math_high:       p=0.??

================================================================
参考: 全員一致率

Task Set        | 3b同種  | 3b真異種 | 7b同種  | 7b異種
----------------|---------|---------|---------|-------
world_consist   | 96.0%  | 57.0%  | 85.0%  | 35.0%
math_elementary | ??.?%  | ??.?%  | ??.?%  | ??.?%
math_middle     | ??.?%  | ??.?%  | ??.?%  | ??.?%
math_high       | ??.?%  | ??.?%  | ??.?%  | ??.?%

================================================================
North Star A-4への接続:
  [実験後に記入]
================================================================
"""
```

---

## 実行順序

```
cd experiments/nca_llm/v5

# 優先度1: 7b条件から実行（最重要）
Step 1: python run_nca_v5.py --task math_elementary --condition 7b_homo
Step 2: python run_nca_v5.py --task math_elementary --condition 7b_het
Step 3: python run_nca_v5.py --task math_middle --condition 7b_homo
Step 4: python run_nca_v5.py --task math_middle --condition 7b_het
Step 5: python run_nca_v5.py --task math_high --condition 7b_homo
Step 6: python run_nca_v5.py --task math_high --condition 7b_het

# 優先度2: 3b条件（補完）
Step 7:  python run_nca_v5.py --task math_elementary --condition 3b_homo
Step 8:  python run_nca_v5.py --task math_elementary --condition 3b_het
Step 9:  python run_nca_v5.py --task math_middle --condition 3b_homo
Step 10: python run_nca_v5.py --task math_middle --condition 3b_het
Step 11: python run_nca_v5.py --task math_high --condition 3b_homo
Step 12: python run_nca_v5.py --task math_high --condition 3b_het

# 分析（優先度1完了後に中間分析も可能）
Step 13: python analyze_v5.py

# コミット
git add results/nca_llm/v5/
git commit -m "data: run MVE-20260404-05 math task diversity experiment"
git push
```

---

## 所要時間の目安

```
優先度1（7b × 6実験）:
  各実験 約45〜60分
  合計: 約4〜6時間

優先度2（3b × 6実験）:
  各実験 約20〜30分
  合計: 約2〜3時間

全体: 約6〜9時間

推奨:
  優先度1を先に走らせて
  中間分析を確認してから
  優先度2を実行する
```

---

## 成功基準（再掲）

```
必須（3つ全て）:
  7b異種 CFR < 7b同種 CFR（math_elementary）
  7b異種 CFR < 7b同種 CFR（math_middle）
  7b異種 CFR < 7b同種 CFR（math_high）

3つ全て → FULL SUCCESS（普遍性確認）
2つ     → PARTIAL（難易度依存の可能性）
1つ以下 → FAIL（world_consistencyのみ特有）
```
