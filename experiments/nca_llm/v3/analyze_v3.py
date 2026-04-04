"""
analyze_v3.py

MVE-20260404-03: Model diversity effect on groupthink.
Compare 4 conditions: 3b homo/hetero, 7b homo/hetero.
"""

import json
import math
import sys
from pathlib import Path

try:
    from scipy.stats import beta as beta_dist, norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Config ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm"

FILES = {
    "3b_homo": BASE / "v2" / "nca_3b_results.jsonl",
    "3b_hetero": BASE / "v3" / "nca_3b_heterogeneous_results.jsonl",
    "7b_homo": BASE / "v2" / "nca_7b_results.jsonl",
    "7b_hetero": BASE / "v1" / "nca_v1_results.jsonl",
}

LABELS = {
    "3b_homo": "NCA 3b homo",
    "3b_hetero": "NCA 3b hetero  <- NEW",
    "7b_homo": "NCA 7b homo",
    "7b_hetero": "NCA 7b hetero",
}


# ── Stats ─────────────────────────────────────────────────────────────────────

def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    if HAS_SCIPY:
        lo = beta_dist.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta_dist.ppf(0.975, k + 1, n - k) if k < n else 1.0
    else:
        z = 1.96
        p = k / n
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        s = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
        lo, hi = max(0, c - s), min(1, c + s)
    return (lo, hi)


def norm_cdf(x: float) -> float:
    if HAS_SCIPY:
        return norm.cdf(x)
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    p_val = 2 * (1 - norm_cdf(abs(z)))
    return (z, p_val)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    seen, deduped = set(), []
    for r in records:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            deduped.append(r)
    return deduped


def analyze_nca(records: list[dict]) -> dict:
    n = len(records)
    n_wrong = n_unan = n_wrong_unan = 0
    for r in records:
        if not r.get("is_correct", False):
            n_wrong += 1
        if r.get("is_unanimous", False):
            n_unan += 1
            if not r.get("is_correct", False):
                n_wrong_unan += 1
    er = n_wrong / n if n else 0
    cfr = n_wrong_unan / n_unan if n_unan else 0
    unan_rate = n_unan / n if n else 0
    er_lo, er_hi = clopper_pearson(n_wrong, n)
    cfr_lo, cfr_hi = clopper_pearson(n_wrong_unan, n_unan)
    return {
        "n": n, "n_wrong": n_wrong, "error_rate": er, "er_ci_lo": er_lo, "er_ci_hi": er_hi,
        "n_unan": n_unan, "n_wrong_unan": n_wrong_unan, "cfr": cfr,
        "cfr_ci_lo": cfr_lo, "cfr_ci_hi": cfr_hi, "unan_rate": unan_rate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    missing = [k for k, v in FILES.items() if not v.exists()]
    if missing:
        print("ERROR: Missing result files:")
        for k in missing:
            print(f"  {k}: {FILES[k]}")
        sys.exit(1)

    stats = {}
    for key, path in FILES.items():
        stats[key] = analyze_nca(load_jsonl(path))

    h = stats["3b_homo"]
    het = stats["3b_hetero"]

    # Primary comparison: CFR(3b hetero) vs CFR(3b homo)
    # z-test: homo CFR vs hetero CFR (homo - hetero, positive if homo > hetero)
    z_main, p_main = two_prop_ztest(h["n_wrong_unan"], h["n_unan"],
                                     het["n_wrong_unan"], het["n_unan"])
    delta_main = (h["cfr"] - het["cfr"]) * 100  # positive = hetero is lower

    # ── Output ────────────────────────────────────────────────────────────

    print("=" * 72)
    print("MVE-20260404-03 Results: Model Diversity vs Groupthink")
    print("=" * 72)

    print(f"{'Condition':<24}| {'Error Rate':<22}| {'CFR':<22}| {'Unanimity':>9}")
    print(f"{'-'*24}|{'-'*22}|{'-'*22}|{'-'*10}")

    for key in ["3b_homo", "3b_hetero", "7b_homo", "7b_hetero"]:
        s = stats[key]
        er_str = f"{s['error_rate']*100:5.1f}% [{s['er_ci_lo']*100:.1f}, {s['er_ci_hi']*100:.1f}]"
        cfr_str = f"{s['cfr']*100:5.1f}% [{s['cfr_ci_lo']*100:.1f}, {s['cfr_ci_hi']*100:.1f}]"
        print(f"{LABELS[key]:<24}| {er_str:<21}| {cfr_str:<21}| {s['unan_rate']*100:8.1f}%")

    print()
    for key in ["3b_homo", "3b_hetero", "7b_homo", "7b_hetero"]:
        s = stats[key]
        print(f"  {key}: wrong_unan={s['n_wrong_unan']}, unan={s['n_unan']}, total={s['n']}")

    print()
    print("=" * 72)
    print("Primary comparison: CFR(3b hetero) vs CFR(3b homo)")
    print(f"  CFR(3b homo):  {h['cfr']*100:.1f}%")
    print(f"  CFR(3b hetero): {het['cfr']*100:.1f}%")
    print(f"  delta: {delta_main:+.1f}pp (homo - hetero)")
    print(f"  z={z_main:.4f}  p={p_main:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_main < 0.05 else 'NO'}")

    print()
    print("=" * 72)

    # Criterion 1: CFR(3b hetero) < CFR(3b homo)
    crit1 = het["cfr"] < h["cfr"]
    print(f"Criterion 1: CFR(3b hetero) < CFR(3b homo 38.5%)")
    print(f"  {het['cfr']*100:.1f}% {'<' if crit1 else '>='} {h['cfr']*100:.1f}%")
    print(f"  -> {'SUCCESS' if crit1 else 'FAIL'}")
    print()

    # Criterion 2: p < 0.05
    crit2 = p_main < 0.05
    print(f"Criterion 2: p < 0.05")
    print(f"  p = {p_main:.4f}")
    print(f"  -> {'SUCCESS' if crit2 else 'FAIL'}")

    print()
    print("=" * 72)

    # Diversity effect overview
    s3h, s3het = stats["3b_homo"], stats["3b_hetero"]
    s7h, s7het = stats["7b_homo"], stats["7b_hetero"]
    print("Diversity effect overview:")
    print(f"  3b: homo {s3h['cfr']*100:.1f}% -> hetero {s3het['cfr']*100:.1f}% "
          f"(delta: {(s3h['cfr']-s3het['cfr'])*100:+.1f}pp)")
    print(f"  7b: homo {s7h['cfr']*100:.1f}% -> hetero {s7het['cfr']*100:.1f}% "
          f"(delta: {(s7h['cfr']-s7het['cfr'])*100:+.1f}pp)")
    print()
    print("Unanimity overview:")
    print(f"  3b: homo {s3h['unan_rate']*100:.1f}% -> hetero {s3het['unan_rate']*100:.1f}%")
    print(f"  7b: homo {s7h['unan_rate']*100:.1f}% -> hetero {s7het['unan_rate']*100:.1f}%")

    print()
    print("=" * 72)

    if crit1 and crit2:
        verdict = "SUCCESS"
    elif crit1 or crit2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"Overall verdict: {verdict}")
    print()
    print("North Star A-4 connection:")
    print("  [Fill in after reviewing results]")
    print("=" * 72)


if __name__ == "__main__":
    main()
