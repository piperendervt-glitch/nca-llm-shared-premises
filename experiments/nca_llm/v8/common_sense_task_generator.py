"""
常識推論タスクジェネレーター
形式A（World ruleなし・CORRECT/INCORRECT）

5カテゴリ × 20問 = 100問
seed=42・決定論的
CORRECT/INCORRECT各50問
"""

import random
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    task_set: str
    task_type: str
    question: str   # "Statement: ..." の形式
    label: bool     # True=CORRECT, False=INCORRECT


def generate_tasks(seed=42):
    random.seed(seed)
    tasks = []

    # ① 自然科学（20問: CORRECT 10問・INCORRECT 10問）
    science_correct = [
        ("水は摂氏0度（1気圧下）で凍る", True),
        ("植物は光合成によって酸素を生成する", True),
        ("光は音よりも速く伝わる", True),
        ("地球は太陽の周りを約365日で公転する", True),
        ("鉄は空気中で酸化すると赤さびが生じる", True),
        ("水素と酸素が結合すると水ができる", True),
        ("ダイヤモンドは炭素でできている", True),
        ("哺乳類は恒温動物である", True),
        ("音は真空中では伝わらない", True),
        ("DNAは遺伝情報を持つ分子である", True),
    ]
    science_incorrect = [
        ("水は摂氏100度以下では沸騰しない（1気圧下）", False),
        ("植物は光合成によって二酸化炭素を生成する", False),
        ("音は光よりも速く伝わる", False),
        ("地球は月の周りを公転している", False),
        ("鉄は水よりも軽い", False),
        ("水素と窒素が結合すると水ができる", False),
        ("ダイヤモンドは鉄でできている", False),
        ("魚類は恒温動物である", False),
        ("音は真空中でも伝わる", False),
        ("RNAは遺伝情報を長期保存する主な分子である", False),
    ]
    for i, (stmt, label) in enumerate(science_correct + science_incorrect):
        tasks.append(Task(
            task_id=f"cs_science_{i:03d}",
            task_set="common_sense",
            task_type="science",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ② 地理・歴史（20問: CORRECT 10問・INCORRECT 10問）
    geo_correct = [
        ("日本は太平洋に面した島国である", True),
        ("富士山は日本最高峰の山である", True),
        ("東京は日本の首都である", True),
        ("太平洋は世界最大の海洋である", True),
        ("ナイル川はアフリカ大陸を流れる", True),
        ("エベレスト山はヒマラヤ山脈に位置する", True),
        ("ブラジルはポルトガル語を公用語とする", True),
        ("オーストラリアは南半球に位置する", True),
        ("アマゾン川は南アメリカ大陸を流れる", True),
        ("カナダはアメリカの北に位置する", True),
    ]
    geo_incorrect = [
        ("日本は大西洋に面した半島である", False),
        ("富士山はアルプス山脈の一部である", False),
        ("大阪は日本の首都である", False),
        ("大西洋は世界最大の海洋である", False),
        ("ナイル川はアジア大陸を流れる", False),
        ("エベレスト山はアンデス山脈に位置する", False),
        ("ブラジルはスペイン語を公用語とする", False),
        ("オーストラリアは北半球に位置する", False),
        ("アマゾン川はアフリカ大陸を流れる", False),
        ("メキシコはアメリカの北に位置する", False),
    ]
    for i, (stmt, label) in enumerate(geo_correct + geo_incorrect):
        tasks.append(Task(
            task_id=f"cs_geo_{i:03d}",
            task_set="common_sense",
            task_type="geography",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ③ 日常生活の常識（20問: CORRECT 10問・INCORRECT 10問）
    daily_correct = [
        ("冷蔵庫は食品の腐敗を遅らせる効果がある", True),
        ("電子レンジは金属を加熱することができない", True),
        ("石鹸は油汚れを落とす効果がある", True),
        ("アルコールは消毒効果がある", True),
        ("ガラスは熱を加えると溶融する", True),
        ("木材は水に浮く", True),
        ("鉄は磁石に引き寄せられる", True),
        ("砂糖は水に溶ける", True),
        ("油は水に溶けない", True),
        ("ゴムは電気を通しにくい", True),
    ]
    daily_incorrect = [
        ("冷蔵庫は食品の腐敗を促進する", False),
        ("電子レンジは金属を安全に加熱できる", False),
        ("石鹸は油汚れを定着させる効果がある", False),
        ("アルコールは細菌を増殖させる効果がある", False),
        ("ガラスは熱を加えても溶融しない", False),
        ("鉛は水に浮く", False),
        ("アルミニウムは磁石に強く引き寄せられる", False),
        ("塩は水に溶けない", False),
        ("油は水に完全に溶ける", False),
        ("銅は電気を通しにくい", False),
    ]
    for i, (stmt, label) in enumerate(daily_correct + daily_incorrect):
        tasks.append(Task(
            task_id=f"cs_daily_{i:03d}",
            task_set="common_sense",
            task_type="daily_life",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ④ 数量・比較（20問: CORRECT 10問・INCORRECT 10問）
    quantity_correct = [
        ("1キログラムは1000グラムである", True),
        ("1時間は60分である", True),
        ("1メートルは100センチメートルである", True),
        ("地球の直径は月の直径より大きい", True),
        ("太陽は地球より大きい", True),
        ("1リットルは1000ミリリットルである", True),
        ("光の速度は秒速約30万キロメートルである", True),
        ("人間の平均体温は約37度である", True),
        ("水の沸点は100度（1気圧）である", True),
        ("1年は12ヶ月である", True),
    ]
    quantity_incorrect = [
        ("1キログラムは100グラムである", False),
        ("1時間は100分である", False),
        ("1メートルは10センチメートルである", False),
        ("月の直径は地球の直径より大きい", False),
        ("地球は太陽より大きい", False),
        ("1リットルは100ミリリットルである", False),
        ("光の速度は秒速約300キロメートルである", False),
        ("人間の平均体温は約20度である", False),
        ("水の沸点は50度（1気圧）である", False),
        ("1年は24ヶ月である", False),
    ]
    for i, (stmt, label) in enumerate(quantity_correct + quantity_incorrect):
        tasks.append(Task(
            task_id=f"cs_quantity_{i:03d}",
            task_set="common_sense",
            task_type="quantity",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ⑤ 因果関係（20問: CORRECT 10問・INCORRECT 10問）
    causal_correct = [
        ("雨が降ると地面が濡れる", True),
        ("運動すると心拍数が上がる", True),
        ("火にかけると水が蒸発する", True),
        ("日光を浴びると影ができる", True),
        ("冷やすと金属は収縮する傾向がある", True),
        ("種を土に植えると水と光があれば発芽する", True),
        ("電流を流すと電球が発光する", True),
        ("気温が下がると水が凍りやすくなる", True),
        ("風が吹くと木の葉が揺れる", True),
        ("重力があるため物体は地面に落下する", True),
    ]
    causal_incorrect = [
        ("雨が降ると地面が乾燥する", False),
        ("運動すると心拍数が下がる", False),
        ("火にかけると水が固まる", False),
        ("日光を浴びると影がなくなる", False),
        ("冷やすと金属は膨張する", False),
        ("種を土に植えるだけで水なしでも必ず発芽する", False),
        ("電流を流すと電球が消える", False),
        ("気温が上がると水が凍りやすくなる", False),
        ("風が吹いても木の葉は動かない", False),
        ("重力がないため物体は地面に落下する", False),
    ]
    for i, (stmt, label) in enumerate(causal_correct + causal_incorrect):
        tasks.append(Task(
            task_id=f"cs_causal_{i:03d}",
            task_set="common_sense",
            task_type="causality",
            question=f"Statement: {stmt}",
            label=label
        ))

    # シャッフルして返す
    random.shuffle(tasks)
    return tasks


if __name__ == "__main__":
    tasks = generate_tasks()
    correct = sum(1 for t in tasks if t.label)
    incorrect = sum(1 for t in tasks if not t.label)
    print(f"Generated {len(tasks)} tasks")
    print(f"CORRECT: {correct} / INCORRECT: {incorrect}")
    types = {}
    for t in tasks:
        types[t.task_type] = types.get(t.task_type, 0) + 1
    for k, v in sorted(types.items()):
        print(f"  {k}: {v}")
    print("\nSample tasks:")
    for t in tasks[:3]:
        print(f"  {t.task_id}: {t.question[:50]}... -> {'CORRECT' if t.label else 'INCORRECT'}")
