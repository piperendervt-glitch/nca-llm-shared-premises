"""
analyze_v5.py

MVE-20260404-05: Diversity effect across task types.
Analyze all 16 conditions (4 task sets x 4 conditions).
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

# WC data from v1-v4
WC_FILES = {
    "3b_homo": BASE / "v2" / "nca_3b_results.jsonl",
    "3b_het":  BASE / "v4" / "nca_3b_true_het_results.jsonl",
    "7b_homo": BASE / "v2" / "nca_7b_results.jsonl",
    "7b_het":  BASE / "v1" / "nca_v1_results.jsonl",
}

# Math data from v5
MATH_TASKS = ["math_elementary", "math_middle", "math_high"]
MATH_PREFIXES = {"math_elementary": "math_elem", "math_middle": "math_mid", "math_high": "math_high"}
CONDITIONS = ["3b_homo", "3b_het", "7b_homo", "7b_het"]


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
    if n == 0:
        return {"n": 0, "n_wrong": 0, "error_rate": 0, "n_unan": 0,
                "n_wrong_unan": 0, "cfr": 0, "unan_rate": 0,
                "er_ci_lo": 0, "er_ci_hi": 0, "cfr_ci_lo": 0, "cfr_ci_hi": 0}
    n_wrong = n_unan = n_wrong_unan = 0
    for r in records:
        if not r.get("is_correct", False):
            n_wrong += 1
        if r.get("is_unanimous", False):
            n_unan += 1
            if not r.get("is_correct", False):
                n_wrong_unan += 1
    er = n_wrong / n
    cfr = n_wrong_unan / n_unan if n_unan else 0
    unan_rate = n_unan / n
    er_lo, er_hi = clopper_pearson(n_wrong, n)
    cfr_lo, cfr_hi = clopper_pearson(n_wrong_unan, n_unan)
    return {
        "n": n, "n_wrong": n_wrong, "error_rate": er, "er_ci_lo": er_lo, "er_ci_hi": er_hi,
        "n_unan": n_unan, "n_wrong_unan": n_wrong_unan, "cfr": cfr,
        "cfr_ci_lo": cfr_lo, "cfr_ci_hi": cfr_hi, "unan_rate": unan_rate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all data
    data = {}  # data[task_set][condition] = stats

    # WC from v1-v4
    data["world_consist"] = {}
    for cond, path in WC_FILES.items():
        recs = load_jsonl(path)
        data["world_consist"][cond] = analyze_nca(recs)

    # Math from v5
    missing = []
    for task in MATH_TASKS:
        prefix = MATH_PREFIXES[task]
        short = task.replace("math_", "")
        data[short] = {}
        for cond in CONDITIONS:
            path = BASE / "v5" / f"{prefix}_{cond}.jsonl"
            if not path.exists():
                missing.append(f"{prefix}_{cond}.jsonl")
                data[short][cond] = analyze_nca([])
            else:
                data[short][cond] = analyze_nca(load_jsonl(path))

    if missing:
        print(f"WARNING: {len(missing)} result files missing:")
        for m in missing:
            print(f"  {m}")
        print()

    # ── Output ────────────────────────────────────────────────────────────

    task_labels = ["world_consist", "elementary", "middle", "high"]
    cond_labels = ["3b_homo", "3b_het", "7b_homo", "7b_het"]

    print("=" * 80)
    print("MVE-20260404-05 Results: Diversity Effect by Task Type")
    print("=" * 80)

    # CFR table
    print()
    print("CFR by condition and task set:")
    print()
    hdr = f"{'Task Set':<16}| {'3b homo':>8} | {'3b het':>8} | {'7b homo':>8} | {'7b het':>8} | {'7b delta':>9}"
    print(hdr)
    print(f"{'-'*16}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}")

    criteria_met = 0
    for ts in task_labels:
        d = data[ts]
        vals = []
        for c in cond_labels:
            s = d.get(c, {"cfr": 0, "n": 0})
            if s["n"] == 0:
                vals.append("   N/A")
            else:
                vals.append(f"{s['cfr']*100:6.1f}%")

        # 7b delta
        h7 = d.get("7b_homo", {"cfr": 0, "n_unan": 0})
        het7 = d.get("7b_het", {"cfr": 0, "n_unan": 0})
        if h7["n"] > 0 and het7["n"] > 0:
            delta = (het7["cfr"] - h7["cfr"]) * 100
            direction = "ok" if delta < 0 else "X"
            delta_str = f"{delta:+6.1f}pp {direction}"
            if delta < 0:
                criteria_met += 1
        else:
            delta_str = "     N/A"

        print(f"{ts:<16}| {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {delta_str}")

    print()
    print("=" * 80)

    # Criteria check (math tasks only)
    print("Criteria: 7b het CFR < 7b homo CFR (direction consistent)")
    print()
    math_criteria = 0
    for ts in ["elementary", "middle", "high"]:
        d = data[ts]
        h7 = d.get("7b_homo", {"cfr": 0, "n": 0, "n_unan": 0, "n_wrong_unan": 0})
        het7 = d.get("7b_het", {"cfr": 0, "n": 0, "n_unan": 0, "n_wrong_unan": 0})
        if h7["n"] == 0 or het7["n"] == 0:
            print(f"  {ts}: DATA MISSING")
            continue
        ok = het7["cfr"] < h7["cfr"]
        z, p = two_prop_ztest(h7["n_wrong_unan"], h7["n_unan"],
                               het7["n_wrong_unan"], het7["n_unan"])
        print(f"  {ts}: 7b het {het7['cfr']*100:.1f}% vs 7b homo {h7['cfr']*100:.1f}% "
              f"(p={p:.4f}) -> {'SUCCESS' if ok else 'FAIL'}")
        if ok:
            math_criteria += 1

    print()
    print("=" * 80)

    # Unanimity table
    print()
    print("Unanimity rate by condition and task set:")
    print()
    hdr = f"{'Task Set':<16}| {'3b homo':>8} | {'3b het':>8} | {'7b homo':>8} | {'7b het':>8}"
    print(hdr)
    print(f"{'-'*16}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}")
    for ts in task_labels:
        d = data[ts]
        vals = []
        for c in cond_labels:
            s = d.get(c, {"unan_rate": 0, "n": 0})
            if s["n"] == 0:
                vals.append("   N/A")
            else:
                vals.append(f"{s['unan_rate']*100:6.1f}%")
        print(f"{ts:<16}| {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]}")

    print()
    print("=" * 80)

    # Raw counts
    print()
    print("Raw counts (n_wrong_unan / n_unan / n_total):")
    for ts in task_labels:
        d = data[ts]
        for c in cond_labels:
            s = d.get(c, {"n": 0, "n_unan": 0, "n_wrong_unan": 0})
            if s["n"] > 0:
                print(f"  {ts} {c}: {s['n_wrong_unan']}/{s['n_unan']}/{s['n']}")

    print()
    print("=" * 80)

    # Verdict
    if math_criteria == 3:
        verdict = "FULL SUCCESS"
    elif math_criteria == 2:
        verdict = "PARTIAL (2/3)"
    elif math_criteria == 1:
        verdict = "PARTIAL (1/3)"
    else:
        verdict = "FAIL"
    print(f"Overall verdict: {verdict}")
    print()
    print("North Star A-4 connection:")
    print("  [Fill in after reviewing results]")
    print("=" * 80)


if __name__ == "__main__":
    main()
