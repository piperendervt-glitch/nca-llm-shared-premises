# 実装指示書: MVE-20260404-03
# NCA-LLM v3: モデル多様性によるグループシンク抑制

## 概要

MVE-20260404-03（ADVANCE済み）の実験を実装する。

北極星: A-4
Claim: 「同じモデルサイズでも異種モデルを組み合わせると
        同種モデルよりグループシンク（CFR）が低くなる」

---

## 新規実験（1つのみ）

```
条件2: 3b異種NCA
  モデル: qwen2.5:3b / llama3.2:3b / gemma2:2b
  roles: Solver（qwen）/ Verifier（llama3.2）/ Critic（gemma2）
  steps: 3
  agree: [30, 80, 80]
  aggregation: Counter多数決（simple majority）
  tasks: world_consistency 100問
  出力: results/nca_llm/v3/nca_3b_heterogeneous_results.jsonl
```

## 流用するデータ（新規実験不要）

```
条件1（3b同種）: results/nca_llm/v2/nca_3b_results.jsonl
  CFR: 38.5%（比較ベースライン）

条件3（7b同種）: results/nca_llm/v2/nca_7b_results.jsonl
  CFR: 30.6%（参考）

条件4（7b異種）: results/nca_llm/v1/nca_v1_results.jsonl
  CFR: 14.3%（参考）
```

---

## ファイル構成

```
nca-llm-shared-premises/
├── experiments/nca_llm/v3/
│   ├── run_nca_v3_3b_het.py   # 3b異種NCA実験
│   ├── analyze_v3.py           # 全条件の分析
│   └── task_generator.py       # v1からコピー
└── results/nca_llm/v3/
    └── nca_3b_heterogeneous_results.jsonl
```

---

## 実装仕様

### run_nca_v3_3b_het.py

```python
# 3b異種NCA実験
# v2のrun_nca_v2_3b.pyをベースに
# モデルを3種類に変更するだけ

# 設定:
models = [
    "qwen2.5:3b",    # Solver
    "llama3.2:3b",   # Verifier
    "gemma2:2b",     # Critic（注: 2bモデル）
]
roles = ["Solver", "Verifier", "Critic"]
steps = 3
agree = [30, 80, 80]
aggregation = "simple_majority"  # Counter

# 出力: results/nca_llm/v3/nca_3b_heterogeneous_results.jsonl

# レコード構造（v2と同一）:
{
    "task_id": str,
    "task_set": "world_consistency",
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
    "models_used": {
        "solver":   "qwen2.5:3b",
        "verifier": "llama3.2:3b",
        "critic":   "gemma2:2b"
    },
    "elapsed_sec": float
}
```

### analyze_v3.py

```python
# MVE-20260404-03の成功基準に従って分析
# 全4条件を比較

# 読み込むファイル:
# 条件1（3b同種）: results/nca_llm/v2/nca_3b_results.jsonl
# 条件2（3b異種）: results/nca_llm/v3/nca_3b_heterogeneous_results.jsonl
# 条件3（7b同種）: results/nca_llm/v2/nca_7b_results.jsonl
# 条件4（7b異種）: results/nca_llm/v1/nca_v1_results.jsonl

# 出力フォーマット:
"""
================================================================
MVE-20260404-03 Results: Model Diversity vs Groupthink
================================================================
条件              | エラー率  | CFR            | 全員一致率
------------------|-----------|----------------|----------
NCA 3b同種        | 39.0%    | 38.5% [?,?]   | 96.0%
NCA 3b異種 ← NEW | ??.?%    | ??.?% [?,?]   | ??.?%
NCA 7b同種        | 34.0%    | 30.6% [?,?]   | 85.0%
NCA 7b異種        | 28.0%    | 14.3% [?,?]   | 35.0%
================================================================
主要比較: CFR(3b異種) vs CFR(3b同種 38.5%)
  delta: ±??.?pp
  z統計量: ???  p値: 0.??
  有意差（p<0.05）: YES / NO

================================================================
必須指標1: CFR(3b異種) < CFR(3b同種 38.5%)
  → SUCCESS / FAIL

必須指標2: p < 0.05
  → SUCCESS / FAIL

================================================================
多様性効果の全体像:
  3b: 同種38.5% → 異種??.?%（差: ??.?pp）
  7b: 同種30.6% → 異種14.3%（差: 16.3pp）

================================================================
総合判定: SUCCESS / PARTIAL / FAIL
North Star A-4への接続:
  [実験後に記入]
================================================================
"""
```

---

## 実装の注意事項

```
① モデルの確認
  ollama list | grep -E "qwen2.5:3b|llama3.2:3b|gemma2:2b"
  3つ全て存在することを確認

  なければ:
  ollama pull llama3.2:3b
  ollama pull gemma2:2b

② gemma2:2bは2bモデル
  「3b異種」ではなく「3bクラス異種」として扱う
  Limitationsに記載する

③ task_generator.pyをv2からコピー
  cp experiments/nca_llm/v2/task_generator.py
     experiments/nca_llm/v3/task_generator.py

④ models_usedフィールドを必ず記録
  どのモデルがどのロールを担当したか
  再現性のために重要

⑤ resume機能を実装
  既存のjsonlがあれば続きから実行

⑥ 実験完了後にverify_results.pyを実行
```

---

## 実行順序

```
cd experiments/nca_llm/v3

Step 1: モデルの確認
  ollama list

Step 2: 3b異種NCA実験
  python run_nca_v3_3b_het.py

Step 3: 分析（全4条件）
  python analyze_v3.py

Step 4: コミット
  git add results/nca_llm/v3/
  git commit -m "data: run MVE-20260404-03 heterogeneous 3b NCA"
  git push
```

---

## 所要時間の目安

```
3b異種NCA × 100問: 約30〜60分
分析: 数分
合計: 約1時間
```

---

## 成功基準（再掲）

```
必須1: CFR(3b異種) < CFR(3b同種 38.5%)
必須2: p < 0.05

両方成立 → SUCCESS → MVEシートに結果記入
片方のみ → PARTIAL → 失敗パターンとして記録
両方不成立 → FAIL → 次のREVISE/PIVOTを検討
```
