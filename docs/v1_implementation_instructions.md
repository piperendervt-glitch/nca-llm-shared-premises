# 実装指示書: MVE-20260404-01
# NCA-LLM v1: グループシンクの存在確認

## 概要

MVE-20260404-01（ADVANCE済み）の実験を実装する。

北極星: A-4
Claim: 「LLMノードを3つNCA構造で接続すると
        world_consistencyタスクでグループシンクが
        単一LLMより多く発生する」

---

## 実験の構成

### 実験1: 単一エージェント（3モデル個別）

```
モデル1: qwen2.5:7b
モデル2: llama3:latest
モデル3: mistral:7b

各モデルが独立してworld_consistencyを解く
相互参照なし・1ステップのみ
```

### 実験2: NCA-LLM（3ノード）

```
qwen2.5:7b（Solver）
llama3:latest（Verifier）
mistral:7b（Critic）

NCA構造（steps=3）で審議
agree=[30, 80, 80]
集約: 単純多数決（Counter）
```

### タスク

```
world_consistency: 100問
ファイル: nca-llm-experiment/data/world_consistency_tasks.jsonl
          または同等のタスクファイル
```

---

## 実装仕様

### ファイル構成

```
nca-llm-shared-premises/
├── experiments/nca_llm/v1/
│   ├── run_single_agent.py   # 単一エージェント実験
│   ├── run_nca_v1.py         # NCA-LLM実験
│   └── analyze_v1.py         # 結果分析
└── results/nca_llm/v1/
    ├── single_qwen.jsonl
    ├── single_llama3.jsonl
    ├── single_mistral.jsonl
    └── nca_v1_results.jsonl
```

### run_single_agent.py の仕様

```python
# 単一エージェント実験
# 各モデルを個別に実行

# 入力: タスクファイル・モデル名
# 出力: results/nca_llm/v1/single_{model}.jsonl

# 各レコードのフィールド:
{
    "task_id": "wc_001",
    "task_set": "world_consistency",
    "task_type": str,
    "question": str,
    "label": bool,          # True=CORRECT, False=INCORRECT
    "model": str,           # モデル名
    "decision": str,        # "CORRECT" or "INCORRECT"
    "confidence": float,    # 0.0〜1.0
    "reasoning": str,
    "is_correct": bool,     # decision == label_str
    "elapsed_sec": float
}

# プロンプト（v9と同じ形式）:
system_prompt = """You are evaluating whether a statement is correct.
Output your response in JSON format:
{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}"""

# 実行:
# python run_single_agent.py --model qwen2.5:7b
# python run_single_agent.py --model llama3:latest
# python run_single_agent.py --model mistral:7b
```

### run_nca_v1.py の仕様

```python
# NCA-LLM実験（3ノード・steps=3）
# nca-llm-experimentのnca_network_v7.pyを参考に実装

# 設定:
models = ["qwen2.5:7b", "llama3:latest", "mistral:7b"]
roles = ["Solver", "Verifier", "Critic"]  # 固定
steps = 3
agree = [30, 80, 80]

# 集約: 単純多数決（System Bと同じ）
# Counter(votes).most_common(1)[0][0]

# 各レコードのフィールド:
{
    "task_id": str,
    "task_set": "world_consistency",
    "task_type": str,
    "question": str,
    "label": bool,
    "prediction": str,        # 最終verdict
    "is_correct": bool,
    "vote_distribution": {    # 最終ステップの票数
        "CORRECT": int,
        "INCORRECT": int
    },
    "is_unanimous": bool,     # 全員一致かどうか
    "node_outputs": {         # 最終ステップの各ノード出力
        "solver": {"decision": str, "confidence": float},
        "verifier": {"decision": str, "confidence": float},
        "critic": {"decision": str, "confidence": float}
    },
    "steps_data": [...],      # 全ステップのデータ
    "elapsed_sec": float
}

# 実行:
# python run_nca_v1.py
```

### analyze_v1.py の仕様

```python
# MVE-20260404-01の成功基準に従って分析

# 必須指標1: 全体エラー率の比較
# 単一エージェント（3モデル）vs NCA-LLM

# 必須指標2: CFR（NCA-LLMのみ）
# CFR = wrong_unanimous / total_unanimous

# 参考指標: 全員一致率
# unanimity_rate = total_unanimous / n_total

# 95% CI: Clopper-Pearson
# z検定: 単一エージェント平均 vs NCA-LLM

# 出力フォーマット:
"""
================================================================
MVE-20260404-01 Results
================================================================
単一エージェント エラー率:
  qwen2.5:7b:  ??.?% [??.?, ??.?]
  llama3:      ??.?% [??.?, ??.?]
  mistral:7b:  ??.?% [??.?, ??.?]
  平均:        ??.?%

NCA-LLM（3ノード・steps=3）:
  エラー率:    ??.?% [??.?, ??.?]
  CFR:         ??.?% [??.?, ??.?]
  全員一致率:  ??.?%

================================================================
必須指標1: NCA エラー率 vs 単一エージェント平均
  delta: ±??.?pp  p=0.??
  → SUCCESS / FAIL

必須指標2: CFR > 20%
  CFR = ??.?%
  → SUCCESS / FAIL

参考指標: 全員一致率 > 50%
  unanimity_rate = ??.?%
  → SUCCESS / FAIL

================================================================
旧データ（参照値・探索フェーズ）との比較:
  旧CFR平均（v2〜v4）: 48.2%
  新CFR:               ??.?%

================================================================
総合判定: SUCCESS / FAIL / PARTIAL
North Star A-4への接続:
  [実験後に記入]
================================================================
"""
```

---

## 実装の注意事項

```
① タスクファイルの場所を確認
  nca-llm-experimentから world_consistencyタスクを
  共有または再生成する

② httpxでOllamaに接続
  URL: http://localhost:11434/api/chat
  timeout: 60秒

③ 進捗表示
  print every 10 tasks

④ resume機能
  既存のjsonlがあれば続きから実行

⑤ verify_results.pyの実行
  実験完了後に必ず実行する
  重複・フィールド確認・CFR計算の検証

⑥ コミットメッセージ
  "experiment: run MVE-20260404-01 v1 groupthink detection"
```

---

## 実行順序

```
Step 1: タスクファイルを準備
Step 2: python run_single_agent.py --model qwen2.5:7b
Step 3: python run_single_agent.py --model llama3:latest
Step 4: python run_single_agent.py --model mistral:7b
Step 5: python run_nca_v1.py
Step 6: python analyze_v1.py
Step 7: 結果をMVEシートに記入
Step 8: commit & push
```

---

## 成功基準（再掲）

```
必須1: NCA-LLM エラー率 > 単一エージェント3モデル平均
必須2: CFR（NCA-LLM）> 20%
参考:  全員一致率 > 50%

両必須指標を満たす → SUCCESS → MVEシートを更新
どちらかが未達   → FAIL → 失敗パターンとして記録
```
