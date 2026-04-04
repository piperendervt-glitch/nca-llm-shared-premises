# 実装指示書: MVE-20260404-02
# NCA-LLM v2: グループシンクのモデルサイズ依存性

## 概要

MVE-20260404-02（ADVANCE済み）の実験を実装する。

北極星: A-4
Claim: 「NCA構造におけるグループシンクの発生率は
        モデルサイズに依存する：
        qwen2.5:3b × 3ノードでは高いCFRを示すが
        qwen2.5:7b × 3ノードでは低いCFRにとどまる」

---

## 実験の構成（4条件）

| 条件 | 内容 | 対応 |
|------|------|------|
| 条件1 | 単一エージェント qwen2.5:3b | 新規実験 |
| 条件2 | NCA-LLM qwen2.5:3b × 3ノード | 新規実験 |
| 条件3 | 単一エージェント qwen2.5:7b | 流用（v1済み） |
| 条件4 | NCA-LLM qwen2.5:7b × 3ノード（同種） | 新規実験 |

---

## ファイル構成

```
nca-llm-shared-premises/
├── experiments/nca_llm/v2/
│   ├── run_single_agent_v2.py   # 条件1: qwen2.5:3b単一
│   ├── run_nca_v2_3b.py         # 条件2: qwen2.5:3b × 3ノード
│   ├── run_nca_v2_7b.py         # 条件4: qwen2.5:7b × 3ノード
│   ├── analyze_v2.py            # 全4条件の分析
│   └── task_generator.py        # v1からコピー
└── results/nca_llm/v2/
    ├── single_qwen3b.jsonl      # 条件1の結果
    ├── nca_3b_results.jsonl     # 条件2の結果
    └── nca_7b_results.jsonl     # 条件4の結果
    # 条件3はv1/single_qwen.jsonlを参照
```

---

## 各スクリプトの仕様

### run_single_agent_v2.py（条件1）

```python
# v1のrun_single_agent.pyと同じ実装
# モデルを qwen2.5:3b に変更するだけ

# 実行:
# cd experiments/nca_llm/v2
# python run_single_agent_v2.py --model qwen2.5:3b

# 出力: results/nca_llm/v2/single_qwen3b.jsonl

# レコード構造（v1と同一）:
{
    "task_id": str,
    "task_set": "world_consistency",
    "task_type": str,
    "question": str,
    "label": bool,
    "model": "qwen2.5:3b",
    "decision": str,
    "confidence": float,
    "reasoning": str,
    "is_correct": bool,
    "elapsed_sec": float
}
```

### run_nca_v2_3b.py（条件2）

```python
# qwen2.5:3b × 3ノード（同種）のNCA実験
# 全ノードが同じモデル（qwen2.5:3b）を使用

# 設定:
models = ["qwen2.5:3b", "qwen2.5:3b", "qwen2.5:3b"]
roles = ["Solver", "Verifier", "Critic"]
steps = 3
agree = [30, 80, 80]
aggregation = "simple_majority"  # Counter

# 出力: results/nca_llm/v2/nca_3b_results.jsonl

# レコード構造（v1と同一）:
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
    "elapsed_sec": float
}

# 実行:
# python run_nca_v2_3b.py
```

### run_nca_v2_7b.py（条件4）

```python
# qwen2.5:7b × 3ノード（同種）のNCA実験
# 全ノードが同じモデル（qwen2.5:7b）を使用

# 設定:
models = ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"]
roles = ["Solver", "Verifier", "Critic"]
steps = 3
agree = [30, 80, 80]
aggregation = "simple_majority"  # Counter

# 出力: results/nca_llm/v2/nca_7b_results.jsonl

# レコード構造: run_nca_v2_3bと同一
```

### analyze_v2.py（分析）

```python
# MVE-20260404-02の成功基準に従って分析

# 読み込むファイル:
# 条件1: results/nca_llm/v2/single_qwen3b.jsonl
# 条件2: results/nca_llm/v2/nca_3b_results.jsonl
# 条件3: results/nca_llm/v1/single_qwen.jsonl  ← 流用
# 条件4: results/nca_llm/v2/nca_7b_results.jsonl

# 計算する指標:
# - エラー率（全条件）
# - CFR（NCA条件のみ）
# - 全員一致率（NCA条件のみ）
# - 95% CI（Clopper-Pearson）
# - z検定: CFR(3b) vs CFR(7b)

# 出力フォーマット:
"""
================================================================
MVE-20260404-02 Results: Groupthink vs Model Size
================================================================
条件              | エラー率        | CFR            | 全員一致率
------------------|-----------------|----------------|----------
単一 qwen2.5:3b  | ??.?% [?,?]     | N/A            | N/A
NCA  qwen2.5:3b  | ??.?% [?,?]     | ??.?% [?,?]   | ??.?%
単一 qwen2.5:7b  | ??.?% [?,?]     | N/A            | N/A
NCA  qwen2.5:7b  | ??.?% [?,?]     | ??.?% [?,?]   | ??.?%
================================================================
CFR比較（3b vs 7b）:
  CFR(3b): ??.?%  CFR(7b): ??.?%
  delta: ±??.?pp
  z統計量: ???  p値: 0.??
  有意差（p<0.05）: YES / NO

================================================================
必須指標1: CFR(3b) > CFR(7b) かつ p<0.05
  → SUCCESS / FAIL

必須指標2: CFR(3b NCA) > 30%
  → SUCCESS / FAIL

必須指標3: CFR(7b NCA) < CFR(3b NCA)
  → SUCCESS / FAIL

================================================================
旧データ（参照値）との比較:
  旧3b CFR平均（v2〜v4・異種）: 46〜51%
  新3b CFR（同種qwen2.5:3b）:   ??.?%

================================================================
総合判定: FULL SUCCESS / PARTIAL / FAIL

North Star A-4への接続:
  [実験後に記入]
================================================================
"""
```

---

## 実装の注意事項

```
① task_generator.pyをv1からv2にコピー
  cp experiments/nca_llm/v1/task_generator.py
     experiments/nca_llm/v2/task_generator.py

② qwen2.5:3bがOllamaで利用可能か確認
  ollama list | grep qwen2.5:3b
  なければ: ollama pull qwen2.5:3b

③ 同種3ノードの実装確認
  全ノードが同じモデルを使用
  役割（Solver/Verifier/Critic）は維持
  プロンプトの違いで役割を区別

④ resume機能を実装
  既存のjsonlがあれば続きから実行

⑤ 進捗表示
  print every 10 tasks

⑥ 実験完了後に verify_results.py を実行
  重複確認・CFR計算の検証
```

---

## 実行順序

```
cd experiments/nca_llm/v2

Step 1: qwen2.5:3bが利用可能か確認
  ollama list

Step 2: 条件1（単一 qwen2.5:3b）
  python run_single_agent_v2.py --model qwen2.5:3b

Step 3: 条件2（NCA qwen2.5:3b × 3）
  python run_nca_v2_3b.py

Step 4: 条件4（NCA qwen2.5:7b × 3）
  python run_nca_v2_7b.py

Step 5: 分析（4条件まとめて）
  python analyze_v2.py

Step 6: コミット
  git add results/nca_llm/v2/
  git commit -m "data: run MVE-20260404-02 model-size groupthink experiment"
  git push
```

---

## 所要時間の目安

```
条件1（単一 3b × 100問）: 約30分
条件2（NCA 3b × 100問）:  約1〜1.5時間
条件4（NCA 7b × 100問）:  約1〜2時間
分析:                     数分

合計: 約2.5〜4時間
```

---

## 成功基準（再掲）

```
必須1: CFR(3b) > CFR(7b)（p<0.05）
必須2: CFR(3b NCA) > 30%
必須3: CFR(7b NCA) < CFR(3b NCA)

全て成立 → FULL SUCCESS → analyze_v2.pyの結果をMVEシートに記入
一部成立 → PARTIAL → 失敗パターンとして記録
全て不成立 → FAIL → 次のREVISE/PIVOTを検討
```
