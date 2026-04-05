"""
analyze_v5_extra.py

Combine original 75 + extra 37 math_middle results (n=112)
and run Fisher's exact test on split accuracy.
"""

import json
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v5"


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

    # 95% CI (Clopper-Pearson) for split accuracy
    if n_split > 0:
        ci_lo, ci_hi = proportion_confint(n_split_correct, n_split, alpha=0.05, method="beta")
    else:
        ci_lo, ci_hi = 0, 0

    return {
        "n_total": n_total, "n_unan": n_unan, "n_wrong_unan": n_wrong_unan,
        "cfr": cfr, "uni_rate": uni_rate,
        "n_split": n_split, "n_split_correct": n_split_correct,
        "split_acc": split_acc, "ci_lo": ci_lo * 100, "ci_hi": ci_hi * 100,
        "ovr_acc": ovr_acc,
    }


def main():
    # Load and combine
    homo_orig = load_jsonl(RESULTS_DIR / "math_mid_7b_homo.jsonl")
    homo_extra = load_jsonl(RESULTS_DIR / "math_mid_7b_homo_extra.jsonl")
    het_orig = load_jsonl(RESULTS_DIR / "math_mid_7b_het.jsonl")
    het_extra = load_jsonl(RESULTS_DIR / "math_mid_7b_het_extra.jsonl")

    homo_all = homo_orig + homo_extra
    het_all = het_orig + het_extra

    s_homo_orig = stats(homo_orig)
    s_het_orig = stats(het_orig)
    s_homo = stats(homo_all)
    s_het = stats(het_all)

    print("=" * 64)
    print("MVE-05 補完実験: math_middle split精度の統計的検証")
    print("=" * 64)
    print(f"結合後のサンプル（既存{len(homo_orig)}問 + 追加{len(homo_extra)}問 = {len(homo_all)}問）:")
    print()

    hdr = f"{'条件':<12} | {'n_total':>7} | {'n_unani':>7} | {'CFR':>7} | {'n_split':>7} | {'split_acc':>9} | {'95%CI':>14}"
    print(hdr)
    print("-" * 12 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 9 + "-+-" + "-" * 14)

    for label, s in [("7b_homo", s_homo), ("7b_het", s_het)]:
        ci_str = f"[{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]"
        print(f"{label:<12} | {s['n_total']:>7} | {s['n_unan']:>7} | {s['cfr']:>6.1f}% | {s['n_split']:>7} | {s['split_acc']:>8.1f}% | {ci_str:>14}")

    print()

    # Fisher's exact test on split accuracy
    # 2x2 table: [correct_homo, wrong_homo], [correct_het, wrong_het]
    a = s_homo["n_split_correct"]
    b = s_homo["n_split"] - a
    c = s_het["n_split_correct"]
    d = s_het["n_split"] - c

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table)

    print(f"Fisher's exact test (split_acc):")
    print(f"  2x2 table: homo [{a} correct, {b} wrong] vs het [{c} correct, {d} wrong]")
    print(f"  Odds ratio: {odds_ratio:.2f}")
    print(f"  p値: {p_value:.4f}")
    print(f"  有意（p<0.05）: {'YES' if p_value < 0.05 else 'NO'}")
    print()

    print("=" * 64)
    print("既存データ（75問）との比較:")
    print(f"  7b_homo split_acc: {s_homo_orig['split_acc']:.1f}%（n={s_homo_orig['n_split']}）"
          f" → {s_homo['split_acc']:.1f}%（n={s_homo['n_split']}）")
    print(f"  7b_het  split_acc: {s_het_orig['split_acc']:.1f}%（n={s_het_orig['n_split']}）"
          f" → {s_het['split_acc']:.1f}%（n={s_het['n_split']}）")
    print()

    print("=" * 64)
    confirmed = p_value < 0.05 and s_homo["split_acc"] > s_het["split_acc"]
    print("Claim判定:")
    print("  「7b同種のsplit精度 > 7b異種のsplit精度」")
    print(f"  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    print()

    print("=" * 64)
    print("A-4への接続:")
    print("  [結果後に記入]")
    print("=" * 64)


if __name__ == "__main__":
    main()
