"""
analyze_v6.py

MVE-20260404-06: Compare diversity effect on logic vs math tasks.
"""

import json
from pathlib import Path

RESULTS_V6 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v6"
RESULTS_V5 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v5"


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stats(rows):
    n_total = len(rows)
    n_unan = sum(1 for r in rows if r["is_unanimous"])
    n_wrong_unan = sum(1 for r in rows if r["is_unanimous"] and not r["is_correct"])
    n_split = n_total - n_unan
    n_split_correct = sum(1 for r in rows if not r["is_unanimous"] and r["is_correct"])
    n_correct = sum(1 for r in rows if r["is_correct"])

    cfr = n_wrong_unan / n_unan * 100 if n_unan else 0
    uni_rate = n_unan / n_total * 100 if n_total else 0
    split_acc = n_split_correct / n_split * 100 if n_split else 0
    ovr_acc = n_correct / n_total * 100 if n_total else 0

    return {
        "n_total": n_total, "n_unan": n_unan, "n_wrong_unan": n_wrong_unan,
        "cfr": cfr, "uni_rate": uni_rate,
        "n_split": n_split, "n_split_correct": n_split_correct,
        "split_acc": split_acc, "ovr_acc": ovr_acc,
    }


def main():
    # Load logic results
    logic_homo = load_jsonl(RESULTS_V6 / "logic_7b_homo.jsonl")
    logic_het = load_jsonl(RESULTS_V6 / "logic_7b_het.jsonl")

    # Load math_middle combined (original + extra)
    math_homo = (load_jsonl(RESULTS_V5 / "math_mid_7b_homo.jsonl") +
                 load_jsonl(RESULTS_V5 / "math_mid_7b_homo_extra.jsonl"))
    math_het = (load_jsonl(RESULTS_V5 / "math_mid_7b_het.jsonl") +
                load_jsonl(RESULTS_V5 / "math_mid_7b_het_extra.jsonl"))

    s_lh = stats(logic_homo)
    s_le = stats(logic_het)
    s_mh = stats(math_homo)
    s_me = stats(math_het)

    print("=" * 64)
    print("MVE-20260404-06 Results: Logic vs Math Diversity Effect")
    print("=" * 64)
    print()

    hdr = f"{'Task x Condition':<22} | {'n_tot':>5} | {'n_uni':>5} | {'n_wu':>4} | {'CFR':>7} | {'uni%':>6} | {'n_spl':>5} | {'spl_acc':>7} | {'ovr_acc':>7}"
    sep = "-" * 22 + "-+-" + "-" * 5 + "-+-" + "-" * 5 + "-+-" + "-" * 4 + "-+-" + "-" * 7 + "-+-" + "-" * 6 + "-+-" + "-" * 5 + "-+-" + "-" * 7 + "-+-" + "-" * 7
    print(hdr)
    print(sep)

    for label, s in [("logic x 7b_homo", s_lh), ("logic x 7b_het", s_le),
                      ("math_mid x 7b_homo", s_mh), ("math_mid x 7b_het", s_me)]:
        print(f"{label:<22} | {s['n_total']:>5} | {s['n_unan']:>5} | {s['n_wrong_unan']:>4} | {s['cfr']:>6.1f}% | {s['uni_rate']:>5.1f}% | {s['n_split']:>5} | {s['split_acc']:>6.1f}% | {s['ovr_acc']:>6.1f}%")

    print()
    print("=" * 64)
    print("Diversity effect comparison:")
    print()

    logic_cfr_diff = s_lh["cfr"] - s_le["cfr"]
    logic_split_diff = s_le["split_acc"] - s_lh["split_acc"]
    math_cfr_diff = s_mh["cfr"] - s_me["cfr"]
    math_split_diff = s_me["split_acc"] - s_mh["split_acc"]

    print(f"{'':>18} | {'CFR diff':>12} | {'split_acc diff':>14}")
    print(f"{'':>18} | {'(homo-het)':>12} | {'(het-homo)':>14}")
    print("-" * 18 + "-+-" + "-" * 12 + "-+-" + "-" * 14)
    print(f"{'logic_reasoning':>18} | {logic_cfr_diff:>+11.1f}pp | {logic_split_diff:>+13.1f}pp")
    print(f"{'math_middle':>18} | {math_cfr_diff:>+11.1f}pp | {math_split_diff:>+13.1f}pp")
    print(f"{'world_consist':>18} | {'+16.3pp':>12} | {'+17.9pp':>14}")

    print()
    print("=" * 64)
    print("Required criteria:")
    cfr_ok = logic_cfr_diff > 0
    split_ok = logic_split_diff > 0
    print(f"  CFR diff > 0 (positive direction):       {'YES' if cfr_ok else 'NO'} ({logic_cfr_diff:+.1f}pp)")
    print(f"  split_acc diff > 0 (positive direction):  {'YES' if split_ok else 'NO'} ({logic_split_diff:+.1f}pp)")
    print()

    if cfr_ok and split_ok:
        verdict = "SUCCESS"
    elif cfr_ok or split_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print("=" * 64)
    print("Task type breakdown (logic):")
    print()

    task_types = sorted(set(r["task_type"] for r in logic_homo))
    print(f"{'type':<16} | {'homo CFR':>9} | {'het CFR':>9} | {'homo spl_acc':>12} | {'het spl_acc':>12}")
    print("-" * 16 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 12 + "-+-" + "-" * 12)

    for tt in task_types:
        homo_tt = [r for r in logic_homo if r["task_type"] == tt]
        het_tt = [r for r in logic_het if r["task_type"] == tt]
        sh = stats(homo_tt)
        se = stats(het_tt)
        cfr_h = f"{sh['cfr']:.1f}%" if sh['n_unan'] > 0 else "n/a"
        cfr_e = f"{se['cfr']:.1f}%" if se['n_unan'] > 0 else "n/a"
        sa_h = f"{sh['split_acc']:.1f}%" if sh['n_split'] > 0 else "n/a"
        sa_e = f"{se['split_acc']:.1f}%" if se['n_split'] > 0 else "n/a"
        print(f"{tt:<16} | {cfr_h:>9} | {cfr_e:>9} | {sa_h:>12} | {sa_e:>12}")

    print()
    print("=" * 64)
    print(f"Verdict: {verdict}")
    print()
    print("North Star A-4 connection:")
    print("  [Fill in after reviewing results]")
    print("=" * 64)


if __name__ == "__main__":
    main()
