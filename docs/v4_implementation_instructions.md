# 実装指示書: MVE-20260404-04
# NCA-LLM v4: 真の3b異種モデルによる多様性効果の確認

## 概要

MVE-20260404-04（ADVANCE済み）の実験を実装する。

北極星: A-4
Claim: 「真の3bモデルのみで構成した異種NCAは
        同種NCAよりCFRが低くなる」

MVE-03との違い:
  除外: gemma2:2b（2bモデル・交絡あり）
  追加: granite3.1-moe:3b（真の3bモデル）

---

## 新規実験（1つのみ）

```
3b真異種NCA:
  qwen2.5:3b（Solver）
  llama3.2:3b（Verifier）
  granite3.1-moe:3b（Critic）← 新モデル

設定（v2と同一）:
  steps: 3
  agree: [30, 80, 80]
  aggregation: Counter多数決
  tasks: world_consistency 100問

出力: results/nca_llm/v4/nca_3b_true_het_results.jsonl
```

## 流用データ（新規実験不要）

```
3b同種（比較ベースライン）:
  results/nca_llm/v2/nca_3b_results.jsonl
  CFR: 38.5%（unanimous=96, wrong_unanimous=37）

7b同種（参考）:
  results/nca_llm/v2/nca_7b_results.jsonl
  CFR: 30.6%

7b異種（参考）:
  results/nca_llm/v1/nca_v1_results.jsonl
  CFR: 14.3%
```

---

## ファイル構成

```
nca-llm-shared-premises/
├── experiments/nca_llm/v4/
│   ├── run_nca_v4_3b_true_het.py   # 3b真異種NCA実験
│   ├── analyze_v4.py               # 全条件の分析
│   └── task_generator.py           # v1からコピー
└── results/nca_llm/v4/
    └── nca_3b_true_het_results.jsonl
```

---

## 実装仕様

### run_nca_v4_3b_true_het.py

```python
# v3のrun_nca_v3_3b_het.pyをベースに
# gemma2:2b → granite3.1-moe:3b に変更するだけ

models = [
    "qwen2.5:3b",          # Solver
    "llama3.2:3b",         # Verifier
    "granite3.1-moe:3b",   # Critic ← 変更点
]
roles = ["Solver", "Verifier", "Critic"]
steps = 3
agree = [30, 80, 80]
aggregation = "simple_majority"  # Counter

# 出力: results/nca_llm/v4/nca_3b_true_het_results.jsonl

# レコード構造（v3と同一）:
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
        "critic":   "granite3.1-moe:3b"
    },
    "elapsed_sec": float
}
```

### analyze_v4.py

```python
# MVE-20260404-04の成功基準に従って分析
# CFRとグループシンクの関係を明示する

# 読み込むファイル:
# 3b同種:    results/nca_llm/v2/nca_3b_results.jsonl
# 3b真異種:  results/nca_llm/v4/nca_3b_true_het_results.jsonl
# 7b同種:    results/nca_llm/v2/nca_7b_results.jsonl
# 7b異種:    results/nca_llm/v1/nca_v1_results.jsonl

# 出力フォーマット:
"""
================================================================
MVE-20260404-04 Results: True 3b Heterogeneous NCA
================================================================
条件              | n_unani | n_wrong_unani | CFR    | 95%CI   | 全員一致率
------------------|---------|---------------|--------|---------|----------
3b同種（流用）    |    96   |      37       | 38.5%  | [?,?]  |   96.0%
3b真異種（NEW）   |    ??   |      ??       | ??.?%  | [?,?]  |   ??.?%
7b同種（流用）    |    85   |      26       | 30.6%  | [?,?]  |   85.0%
7b異種（流用）    |    35   |       5       | 14.3%  | [?,?]  |   35.0%

================================================================
CFRとグループシンクの関係（トートロジー確認）:
  3b同種:   unanimous=96,  wrong_unanimous=37  → CFR=38.5%
  3b真異種: unanimous=??,  wrong_unanimous=??  → CFR=??.?%

  「全員一致した時の誤り率」の変化:
  → 多様性の効果が「全員一致を防ぐ」のか
    「全員一致した時の精度を上げる」のかを確認

================================================================
必須指標1: CFR(3b真異種) < CFR(3b同種 38.5%)
  → SUCCESS / FAIL

必須指標2: p < 0.05
  z統計量: ???  p値: 0.??
  → SUCCESS / FAIL

参考: エラー率の変化
  3b同種: 39.0% → 3b真異種: ??.?%
  （MVE-03では42%に上昇・今回は改善するか？）

================================================================
多様性効果の全体像（全4条件）:
  3b同種  → 3b真異種: CFR差 ??.?pp
  7b同種  → 7b異種:   CFR差 16.3pp

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
① granite3.1-moe:3bのMoEアーキテクチャ
  論文・結果レポートで明記する:
  「granite3.1-moe:3bはMixture of Expertsアーキテクチャ
   パラメータ数は3bクラスだが内部構造が異なる」

② task_generator.pyをv3からコピー
  cp experiments/nca_llm/v3/task_generator.py
     experiments/nca_llm/v4/task_generator.py

③ resume機能を実装
  既存のjsonlがあれば続きから実行

④ 進捗表示
  print every 10 tasks

⑤ 実験完了後にverify_results.pyを実行
  重複確認・CFR計算の検証
```

---

## 実行順序

```
cd experiments/nca_llm/v4

Step 1: モデルの確認
  ollama list | grep granite

Step 2: 3b真異種NCA実験
  python run_nca_v4_3b_true_het.py

Step 3: 分析（全4条件）
  python analyze_v4.py

Step 4: コミット
  git add results/nca_llm/v4/
  git commit -m "data: run MVE-20260404-04 true 3b heterogeneous NCA"
  git push
```

---

## 所要時間の目安

```
3b真異種NCA × 100問: 約30〜60分
分析: 数分
合計: 約1時間
```

---

## 成功基準（再掲）

```
必須1: CFR(3b真異種) < CFR(3b同種 38.5%)
必須2: p < 0.05

両方成立 → SUCCESS
  多様性効果は3bでも統計的に確認
  MVEシートに結果記入

片方のみ → PARTIAL
  Case A: CFR差あり・p値大
    → n不足・サンプル増加を検討

失敗 → FAIL
  Case B: CFR差なし
    → 「3bでは多様性効果なし」として記録
    → C（ADVANCE）: 7bの発見のみを根拠に前進
  Case C: CFR上昇（逆方向）
    → 「3bでは多様性が有害」として記録
```
