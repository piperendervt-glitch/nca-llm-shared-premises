"""
analyze_v2.py

MVE-20260404-02: Model-size dependency of groupthink.
Compare qwen2.5:3b vs qwen2.5:7b (single-agent and NCA).
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
    "single_3b": BASE / "v2" / "single_qwen3b.jsonl",
    "nca_3b":    BASE / "v2" / "nca_3b_results.jsonl",
    "single_7b": BASE / "v1" / "single_qwen.jsonl",       # reused from v1
    "nca_7b":    BASE / "v2" / "nca_7b_results.jsonl",
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
    z = (k1 / n1 - k2 / n2) / se  # positive if k1/n1 > k2/n2
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


def analyze_single(records: list[dict]) -> dict:
    n = len(records)
    n_wrong = sum(1 for r in records if not r.get("is_correct", False))
    er = n_wrong / n if n else 0
    lo, hi = clopper_pearson(n_wrong, n)
    return {"n": n, "n_wrong": n_wrong, "error_rate": er, "ci_lo": lo, "ci_hi": hi}


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

    s3b = analyze_single(load_jsonl(FILES["single_3b"]))
    n3b = analyze_nca(load_jsonl(FILES["nca_3b"]))
    s7b = analyze_single(load_jsonl(FILES["single_7b"]))
    n7b = analyze_nca(load_jsonl(FILES["nca_7b"]))

    # CFR comparison z-test: CFR(3b) vs CFR(7b)
    z_cfr, p_cfr = two_prop_ztest(n3b["n_wrong_unan"], n3b["n_unan"],
                                   n7b["n_wrong_unan"], n7b["n_unan"])
    delta_cfr = (n3b["cfr"] - n7b["cfr"]) * 100

    # ── Output ────────────────────────────────────────────────────────────
    def fmt_er(s):
        return f"{s['error_rate']*100:5.1f}% [{s['ci_lo']*100:.1f}, {s['ci_hi']*100:.1f}]"

    def fmt_nca_er(s):
        return f"{s['error_rate']*100:5.1f}% [{s['er_ci_lo']*100:.1f}, {s['er_ci_hi']*100:.1f}]"

    def fmt_cfr(s):
        return f"{s['cfr']*100:5.1f}% [{s['cfr_ci_lo']*100:.1f}, {s['cfr_ci_hi']*100:.1f}]"

    print("=" * 72)
    print("MVE-20260404-02 Results: Groupthink vs Model Size")
    print("=" * 72)

    print(f"{'Condition':<22}| {'Error Rate':<22}| {'CFR':<22}| {'Unanimity':>9}")
    print(f"{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}")
    print(f"{'Single qwen2.5:3b':<22}| {fmt_er(s3b):<21}| {'N/A':<21}| {'N/A':>9}")
    print(f"{'NCA    qwen2.5:3b':<22}| {fmt_nca_er(n3b):<21}| {fmt_cfr(n3b):<21}| {n3b['unan_rate']*100:8.1f}%")
    print(f"{'Single qwen2.5:7b':<22}| {fmt_er(s7b):<21}| {'N/A':<21}| {'N/A':>9}")
    print(f"{'NCA    qwen2.5:7b':<22}| {fmt_nca_er(n7b):<21}| {fmt_cfr(n7b):<21}| {n7b['unan_rate']*100:8.1f}%")

    print()
    print(f"  NCA 3b: wrong_unan={n3b['n_wrong_unan']}, unan={n3b['n_unan']}, total={n3b['n']}")
    print(f"  NCA 7b: wrong_unan={n7b['n_wrong_unan']}, unan={n7b['n_unan']}, total={n7b['n']}")

    print()
    print("=" * 72)
    print("CFR comparison (3b vs 7b):")
    print(f"  CFR(3b): {n3b['cfr']*100:.1f}%  CFR(7b): {n7b['cfr']*100:.1f}%")
    print(f"  delta: {delta_cfr:+.1f}pp")
    print(f"  z={z_cfr:.4f}  p={p_cfr:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_cfr < 0.05 else 'NO'}")

    print()
    print("=" * 72)

    # Criterion 1: CFR(3b) > CFR(7b) and p<0.05
    crit1 = n3b["cfr"] > n7b["cfr"] and p_cfr < 0.05
    print(f"Criterion 1: CFR(3b) > CFR(7b) and p<0.05")
    print(f"  CFR(3b)={n3b['cfr']*100:.1f}% vs CFR(7b)={n7b['cfr']*100:.1f}%, p={p_cfr:.4f}")
    print(f"  -> {'SUCCESS' if crit1 else 'FAIL'}")
    print()

    # Criterion 2: CFR(3b NCA) > 30%
    crit2 = n3b["cfr"] > 0.30
    print(f"Criterion 2: CFR(3b NCA) > 30%")
    print(f"  CFR(3b) = {n3b['cfr']*100:.1f}%")
    print(f"  -> {'SUCCESS' if crit2 else 'FAIL'}")
    print()

    # Criterion 3: CFR(7b NCA) < CFR(3b NCA)
    crit3 = n7b["cfr"] < n3b["cfr"]
    print(f"Criterion 3: CFR(7b NCA) < CFR(3b NCA)")
    print(f"  CFR(7b)={n7b['cfr']*100:.1f}% < CFR(3b)={n3b['cfr']*100:.1f}%")
    print(f"  -> {'SUCCESS' if crit3 else 'FAIL'}")

    print()
    print("=" * 72)

    # Reference
    print("Exploration phase reference (3b mixed, v2-v4):")
    print(f"  Old CFR average: 46-51%")
    print(f"  New CFR (3b homogeneous): {n3b['cfr']*100:.1f}%")
    print()

    # Reference: unanimity
    print(f"Reference: unanimity rates")
    print(f"  3b NCA: {n3b['unan_rate']*100:.1f}% ({n3b['n_unan']}/{n3b['n']})")
    print(f"  7b NCA: {n7b['unan_rate']*100:.1f}% ({n7b['n_unan']}/{n7b['n']})")

    print()
    print("=" * 72)

    n_success = sum([crit1, crit2, crit3])
    if n_success == 3:
        verdict = "FULL SUCCESS"
    elif n_success >= 2:
        verdict = "PARTIAL SUCCESS"
    else:
        verdict = "FAIL"
    print(f"Overall verdict: {verdict} ({n_success}/3 criteria met)")
    print()
    print("North Star A-4 connection:")
    print("  [Fill in after reviewing results]")
    print("=" * 72)


if __name__ == "__main__":
    main()
