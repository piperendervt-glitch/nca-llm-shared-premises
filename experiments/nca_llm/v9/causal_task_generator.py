"""
因果推論タスクジェネレーター
形式A（CORRECT/INCORRECT）
5カテゴリ × 20問 = 100問候補
seed=42・決定論的
"""

import random
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    task_set: str
    task_type: str
    question: str
    label: bool


def generate_tasks(seed=42):
    random.seed(seed)
    tasks = []

    # ① 反事実的因果（20問）
    counterfactual = [
        ("雨が降らなければこの洪水は起きなかった", True),
        ("ワクチンがなければその感染症の死者数はさらに多かった", True),
        ("信号が赤でなければその交通事故は起きなかった", True),
        ("冷蔵庫がなければ食品の腐敗はより早く進む", True),
        ("抗生物質がなければその細菌感染症は致命的だった可能性がある", True),
        ("太陽がなければ地球上の生命は存在しなかった", True),
        ("彼が道を渡らなければ車にひかれることはなかった", True),
        ("適切な排水設備がなければ都市の洪水被害は拡大していた", True),
        ("避難指示が早ければ被害者数は減っていた可能性がある", True),
        ("安全装置が作動しなければ事故はより深刻だった", True),
        ("雨が降らなければ必ず農業は成功する", False),
        ("ワクチンがなければすべての人が感染症で死亡する", False),
        ("信号がなければすべての交差点で必ず事故が起きる", False),
        ("冷蔵庫がなければ食品はすべて即座に腐敗する", False),
        ("抗生物質がなければすべての感染症は治療不可能である", False),
        ("太陽がなくても生命は別の形で存在できる（現在の地球環境で）", False),
        ("彼が道を渡らなければ必ず長生きできた", False),
        ("排水設備があれば洪水は絶対に起きない", False),
        ("避難指示が早ければ被害者数は必ずゼロになった", False),
        ("安全装置があれば事故は絶対に起きない", False),
    ]
    for i, (stmt, label) in enumerate(counterfactual):
        tasks.append(Task(
            task_id=f"causal_cf_{i:03d}",
            task_set="causal_reasoning",
            task_type="counterfactual",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ② 多重因果（20問）
    multiple_cause = [
        ("気温と湿度の両方が高い条件が重なると熱中症リスクは大幅に上がる", True),
        ("睡眠不足と過度なストレスが重なると免疫力が低下しやすい", True),
        ("経済格差と教育機会の不平等が組み合わさると社会的流動性が低下する", True),
        ("遺伝的要因と生活習慣の両方が生活習慣病の発症に関わる", True),
        ("土壌の質と降水量の両方が農作物の収穫量に影響する", True),
        ("騒音と光害の両方が都市住民の睡眠の質に悪影響を与えうる", True),
        ("栄養不足と運動不足が重なると筋力低下が進みやすい", True),
        ("高齢化と出生率低下が同時に進むと労働力人口が減少する", True),
        ("気温上昇と海面上昇の両方が沿岸部の洪水リスクを高める", True),
        ("原材料費と人件費の両方の上昇が製品価格の値上がりにつながる", True),
        ("気温が高いだけで必ず熱中症になる", False),
        ("睡眠不足だけで必ず重大な病気になる", False),
        ("経済格差だけが社会的流動性の低下の原因である", False),
        ("生活習慣病は遺伝的要因だけで決まる", False),
        ("降水量だけが農作物の収穫量を決める", False),
        ("騒音だけが睡眠の質に影響する唯一の要因である", False),
        ("栄養不足だけで必ず筋力が低下する", False),
        ("出生率低下だけが労働力人口減少の原因である", False),
        ("気温上昇だけが沿岸部の洪水リスクを決める", False),
        ("原材料費の上昇だけが製品値上がりの原因である", False),
    ]
    for i, (stmt, label) in enumerate(multiple_cause):
        tasks.append(Task(
            task_id=f"causal_mc_{i:03d}",
            task_set="causal_reasoning",
            task_type="multiple_cause",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ③ 遠因・近因（20問）
    distal_proximal = [
        ("長期的な過労が免疫力を低下させ風邪をひきやすい状態をつくった", True),
        ("幼少期の食習慣が成人後の生活習慣病リスクに影響することがある", True),
        ("インフラ整備の遅れが長期的な経済発展の制約になりうる", True),
        ("土壌汚染が食物連鎖を通じて生態系全体に影響を与えうる", True),
        ("教育投資の不足が数十年後の労働力の質に影響しうる", True),
        ("幼少期のトラウマが成人後の精神的健康に影響することがある", True),
        ("温室効果ガスの蓄積が長期的な気候変動を引き起こす", True),
        ("森林伐採が数十年かけて土壌流出と洪水リスクを高める", True),
        ("慢性的な睡眠不足が長期的に認知機能に影響しうる", True),
        ("幼少期の運動習慣が成人後の骨密度に影響することがある", True),
        ("風邪は過労が直接かつ唯一の原因である", False),
        ("幼少期の食習慣だけが成人後の健康状態を決定する", False),
        ("インフラ整備だけが経済発展のすべてを決める", False),
        ("土壌汚染は食物連鎖に一切影響しない", False),
        ("教育投資は翌年すぐに労働力の質に反映される", False),
        ("幼少期のトラウマは成人後に必ず深刻な問題を引き起こす", False),
        ("温室効果ガスは排出後すぐに気候に影響する", False),
        ("森林伐採は翌年すぐに洪水を引き起こす", False),
        ("一晩の睡眠不足が長期的な認知機能低下を引き起こす", False),
        ("幼少期に運動しなかった人は必ず骨密度が低い", False),
    ]
    for i, (stmt, label) in enumerate(distal_proximal):
        tasks.append(Task(
            task_id=f"causal_dp_{i:03d}",
            task_set="causal_reasoning",
            task_type="distal_proximal",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ④ 因果の方向（20問）
    causal_direction = [
        ("経済成長が一般的に教育投資の余裕を生む", True),
        ("需要の増加が供給の増加を促すことがある", True),
        ("人口増加が住宅需要の増加につながる", True),
        ("技術革新が新たな雇用を生み出すことがある", True),
        ("環境規制の強化が企業の技術革新を促すことがある", True),
        ("都市化の進展が公共交通需要を高める", True),
        ("所得の増加が消費の増加につながることが多い", True),
        ("高齢化が医療費の増加をもたらす傾向がある", True),
        ("インターネットの普及が情報流通コストを下げた", True),
        ("農業技術の向上が食料供給の安定に貢献した", True),
        ("教育投資だけが経済成長の原因である", False),
        ("供給の増加が需要を生み出す（常に）", False),
        ("住宅建設が人口増加を引き起こす", False),
        ("技術革新は必ず雇用を破壊する", False),
        ("環境規制は必ず企業の競争力を低下させる", False),
        ("公共交通の整備が都市化を引き起こす（主因として）", False),
        ("消費の増加が必ず所得の増加をもたらす", False),
        ("医療費の増加が高齢化を引き起こす", False),
        ("情報流通コストの低下がインターネットを生み出した", False),
        ("食料供給の安定だけが農業技術向上の結果である", False),
    ]
    for i, (stmt, label) in enumerate(causal_direction):
        tasks.append(Task(
            task_id=f"causal_cd_{i:03d}",
            task_set="causal_reasoning",
            task_type="causal_direction",
            question=f"Statement: {stmt}",
            label=label
        ))

    # ⑤ 相関 vs 因果（20問）
    correlation_causation = [
        ("アイスクリームの売上と溺死者数には相関があるが\nアイスクリームが溺死の原因ではない", True),
        ("靴のサイズと読解力には相関があるが（子どもの年齢が交絡）\n靴のサイズが読解力を決めるわけではない", True),
        ("病院の数が多い地域で死亡率が高いのは\n重病者が病院に集まるためであり病院が死亡を引き起こすわけではない", True),
        ("消防士が多い火事ほど被害が大きいのは\n大きな火事に多くの消防士が派遣されるためである", True),
        ("国の豊かさとインターネット普及率には相関があるが\nどちらが原因かは複雑である", True),
        ("運動する人は健康な傾向があるが\n健康な人が運動できる可能性も考慮が必要である", True),
        ("コーヒーを飲む人に心臓病が多い相関は\n喫煙という交絡変数で説明できる部分がある", True),
        ("社会的つながりが多い人は長寿の傾向があるが\n因果関係の方向は単純ではない", True),
        ("読書量と語彙力には相関があるが\n語彙力が高い人が読書を好む可能性もある", True),
        ("幸福度と収入には相関があるが\n一定水準以上では相関が弱まる傾向がある", True),
        ("相関関係は常に因果関係を意味する", False),
        ("アイスクリームを規制すれば溺死者数が減る", False),
        ("病院の数を減らせば死亡率が下がる", False),
        ("消防士を減らせば火事の被害が減る", False),
        ("インターネット普及率を高めれば国が豊かになる（それだけで）", False),
        ("運動を強制すれば誰でも健康になる", False),
        ("コーヒーを禁止すれば心臓病が減る", False),
        ("社交性を高めれば必ず長寿になる", False),
        ("語彙力を高めれば読書量が増える（主な因果方向として）", False),
        ("収入を増やせば必ず幸福になる", False),
    ]
    for i, (stmt, label) in enumerate(correlation_causation):
        tasks.append(Task(
            task_id=f"causal_cc_{i:03d}",
            task_set="causal_reasoning",
            task_type="correlation_causation",
            question=f"Statement: {stmt}",
            label=label
        ))

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
