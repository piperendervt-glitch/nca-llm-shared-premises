"""
論理推論タスクジェネレーター
形式A（CORRECT/INCORRECT）
world_consistencyと同じパイプラインを使用

5種別 × 20問 = 100問
seed=42・決定論的
"""

import random
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    task_set: str
    task_type: str
    question: str   # world_rule + statement を結合した文字列
    label: bool     # True=CORRECT, False=INCORRECT


def generate_tasks(seed=42):
    random.seed(seed)

    # ① 三段論法（20問）
    syllogism_subjects = [
        ("犬", "哺乳類", "温血動物"),
        ("桜", "植物", "光合成をする生物"),
        ("鉄", "金属", "電気を通す物質"),
        ("東京", "日本の都市", "アジアにある場所"),
        ("サメ", "魚類", "水中で生活する生物"),
        ("バラ", "花", "植物"),
        ("金", "貴金属", "希少な金属"),
        ("雨", "降水", "水循環の一部"),
        ("ピアノ", "鍵盤楽器", "楽器"),
        ("富士山", "火山", "山"),
    ]
    tasks_syl = []
    for i, (A, B, C) in enumerate(syllogism_subjects):
        tasks_syl.append(Task(
            task_id=f"logic_syl_{i*2:03d}",
            task_set="logic_reasoning",
            task_type="syllogism",
            question=(
                f"World rule: すべての{A}は{B}である。"
                f"すべての{B}は{C}である。\n"
                f"Statement: すべての{A}は{C}である。"
            ),
            label=True
        ))
        tasks_syl.append(Task(
            task_id=f"logic_syl_{i*2+1:03d}",
            task_set="logic_reasoning",
            task_type="syllogism",
            question=(
                f"World rule: すべての{A}は{B}である。"
                f"すべての{B}は{C}である。\n"
                f"Statement: すべての{C}は{A}である。"
            ),
            label=False
        ))

    # ② 推移律（20問）
    transitive_items = [
        ("象", "馬", "犬"),
        ("Mount Everest", "富士山", "東京タワー"),
        ("太陽", "地球", "月"),
        ("100", "50", "10"),
        ("大学", "高校", "中学校"),
        ("金", "銀", "銅"),
        ("台風", "大雨", "小雨"),
        ("超特急", "特急", "普通電車"),
        ("億万長者", "会社員", "学生"),
        ("大陸", "国", "都市"),
    ]
    tasks_trans = []
    for i, (A, B, C) in enumerate(transitive_items):
        tasks_trans.append(Task(
            task_id=f"logic_trans_{i*2:03d}",
            task_set="logic_reasoning",
            task_type="transitivity",
            question=(
                f"World rule: {A}は{B}より大きい。"
                f"{B}は{C}より大きい。\n"
                f"Statement: {A}は{C}より大きい。"
            ),
            label=True
        ))
        tasks_trans.append(Task(
            task_id=f"logic_trans_{i*2+1:03d}",
            task_set="logic_reasoning",
            task_type="transitivity",
            question=(
                f"World rule: {A}は{B}より大きい。"
                f"{B}は{C}より大きい。\n"
                f"Statement: {C}は{A}より大きい。"
            ),
            label=False
        ))

    # ③ 対偶（20問）
    contrapositive_rules = [
        ("雨が降る", "地面が濡れる"),
        ("火がある", "煙が出る"),
        ("勉強する", "成績が上がる"),
        ("風邪をひく", "体温が上がる"),
        ("太陽が出る", "影ができる"),
        ("水を加熱する", "蒸発する"),
        ("運動する", "汗をかく"),
        ("停電する", "電気が使えなくなる"),
        ("雪が降る", "道が白くなる"),
        ("花が咲く", "虫が集まる"),
    ]
    tasks_contra = []
    for i, (A, B) in enumerate(contrapositive_rules):
        tasks_contra.append(Task(
            task_id=f"logic_contra_{i*2:03d}",
            task_set="logic_reasoning",
            task_type="contrapositive",
            question=(
                f"World rule: {A}ならば{B}。\n"
                f"Statement: {B}でないならば{A}でない。"
            ),
            label=True
        ))
        tasks_contra.append(Task(
            task_id=f"logic_contra_{i*2+1:03d}",
            task_set="logic_reasoning",
            task_type="contrapositive",
            question=(
                f"World rule: {A}ならば{B}。\n"
                f"Statement: {B}ならば{A}である。"
            ),
            label=False
        ))

    # ④ 排中律・矛盾（20問）
    contradiction_pairs = [
        ("この数は偶数である", "この数は奇数である"),
        ("この部屋は空である", "この部屋に人がいる"),
        ("彼は走っている", "彼は静止している"),
        ("この水は凍っている", "この水は液体である"),
        ("太陽が昇っている", "今は夜中である"),
        ("AはBより大きい", "AはBより小さい"),
        ("この文は真である", "この文は偽である"),
        ("扉は開いている", "扉は閉まっている"),
        ("彼女は結婚している", "彼女は独身である"),
        ("数Xは正である", "数Xは負である"),
    ]
    tasks_contra2 = []
    for i, (A, B) in enumerate(contradiction_pairs):
        tasks_contra2.append(Task(
            task_id=f"logic_contra2_{i*2:03d}",
            task_set="logic_reasoning",
            task_type="contradiction",
            question=(
                f"World rule: 「{A}」と「{B}」は"
                f"同時に成立することはない。\n"
                f"Statement: {A}であれば{B}ではない。"
            ),
            label=True
        ))
        tasks_contra2.append(Task(
            task_id=f"logic_contra2_{i*2+1:03d}",
            task_set="logic_reasoning",
            task_type="contradiction",
            question=(
                f"World rule: 「{A}」と「{B}」は"
                f"同時に成立することはない。\n"
                f"Statement: {A}であり、かつ{B}である。"
            ),
            label=False
        ))

    # ⑤ 複合条件（20問）
    compound_rules = [
        ("雨が降っている", "傘を持っている", "濡れずに外出できる"),
        ("鍵を持っている", "ドアが施錠されていない", "家に入れる"),
        ("材料がある", "料理の技術がある", "料理が作れる"),
        ("チケットがある", "会場に行ける", "コンサートを楽しめる"),
        ("充電されている", "電波がある", "スマートフォンが使える"),
        ("資格がある", "求人がある", "仕事に応募できる"),
        ("種がある", "土がある", "植物を育てられる"),
        ("パスポートがある", "ビザがある", "海外に行ける"),
        ("お金がある", "商品が在庫にある", "買い物ができる"),
        ("練習している", "才能がある", "上達できる"),
    ]
    tasks_compound = []
    for i, (A, B, C) in enumerate(compound_rules):
        tasks_compound.append(Task(
            task_id=f"logic_comp_{i*2:03d}",
            task_set="logic_reasoning",
            task_type="compound",
            question=(
                f"World rule: {A}、かつ{B}ならば{C}。\n"
                f"Statement: {A}であり、{B}である場合、{C}。"
            ),
            label=True
        ))
        tasks_compound.append(Task(
            task_id=f"logic_comp_{i*2+1:03d}",
            task_set="logic_reasoning",
            task_type="compound",
            question=(
                f"World rule: {A}、かつ{B}ならば{C}。\n"
                f"Statement: {A}であれば、{B}がなくても{C}。"
            ),
            label=False
        ))

    # 全タスクを結合してシャッフル
    all_tasks = (tasks_syl + tasks_trans[:20] +
                 tasks_contra[:20] + tasks_contra2[:20] +
                 tasks_compound[:20])
    random.shuffle(all_tasks)

    return all_tasks


if __name__ == "__main__":
    tasks = generate_tasks()
    print(f"Generated {len(tasks)} tasks")

    correct = sum(1 for t in tasks if t.label)
    print(f"Labels: {correct} CORRECT, {len(tasks) - correct} INCORRECT")

    types = {}
    for t in tasks:
        types.setdefault(t.task_type, []).append(t)
    for tt, ts in sorted(types.items()):
        c = sum(1 for t in ts if t.label)
        print(f"  {tt}: {len(ts)} tasks ({c} CORRECT, {len(ts)-c} INCORRECT)")

    print("\nSamples:")
    for task in tasks[:3]:
        print(f"\n{task.task_id}: {task.task_type}")
        print(f"Q: {task.question}")
        print(f"A: {'CORRECT' if task.label else 'INCORRECT'}")
