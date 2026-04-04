"""
analyze_v1.py

MVE-20260404-01: Analyze single-agent and NCA-LLM results.
Calculate error rates, CFR, and check success criteria.
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

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v1"

SINGLE_FILES = {
    "qwen2.5:7b": RESULTS_DIR / "single_qwen.jsonl",
    "llama3": RESULTS_DIR / "single_llama3.jsonl",
    "mistral:7b": RESULTS_DIR / "single_mistral.jsonl",
}
NCA_FILE = RESULTS_DIR / "nca_v1_results.jsonl"


# ── Stats helpers ─────────────────────────────────────────────────────────────

def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    if HAS_SCIPY:
        lo = beta_dist.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta_dist.ppf(0.975, k + 1, n - k) if k < n else 1.0
    else:
        # Wilson interval fallback
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
    """Two-proportion z-test. Returns (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p2 - p1) / se
    p_val = 2 * (1 - norm_cdf(abs(z)))
    return (z, p_val)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    # Deduplicate by task_id (first occurrence)
    seen = set()
    deduped = []
    for r in records:
        tid = r["task_id"]
        if tid not in seen:
            seen.add(tid)
            deduped.append(r)
    return deduped


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_single(records: list[dict]) -> dict:
    n = len(records)
    n_wrong = sum(1 for r in records if not r.get("is_correct", False))
    error_rate = n_wrong / n if n else 0
    lo, hi = clopper_pearson(n_wrong, n)
    return {"n": n, "n_wrong": n_wrong, "error_rate": error_rate, "ci_lo": lo, "ci_hi": hi}


def analyze_nca(records: list[dict]) -> dict:
    n = len(records)
    n_wrong = 0
    n_unanimous = 0
    n_wrong_unanimous = 0

    for r in records:
        is_correct = r.get("is_correct", False)
        if not is_correct:
            n_wrong += 1

        is_unanimous = r.get("is_unanimous", False)
        if is_unanimous:
            n_unanimous += 1
            if not is_correct:
                n_wrong_unanimous += 1

    error_rate = n_wrong / n if n else 0
    cfr = n_wrong_unanimous / n_unanimous if n_unanimous else 0
    unanimity_rate = n_unanimous / n if n else 0

    er_lo, er_hi = clopper_pearson(n_wrong, n)
    cfr_lo, cfr_hi = clopper_pearson(n_wrong_unanimous, n_unanimous)

    return {
        "n": n,
        "n_wrong": n_wrong,
        "error_rate": error_rate,
        "er_ci_lo": er_lo,
        "er_ci_hi": er_hi,
        "n_unanimous": n_unanimous,
        "n_wrong_unanimous": n_wrong_unanimous,
        "cfr": cfr,
        "cfr_ci_lo": cfr_lo,
        "cfr_ci_hi": cfr_hi,
        "unanimity_rate": unanimity_rate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check files exist
    missing = []
    for name, path in SINGLE_FILES.items():
        if not path.exists():
            missing.append(str(path))
    if not NCA_FILE.exists():
        missing.append(str(NCA_FILE))
    if missing:
        print("ERROR: Missing result files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun the experiments first:")
        print("  python run_single_agent.py --model qwen2.5:7b")
        print("  python run_single_agent.py --model llama3:latest")
        print("  python run_single_agent.py --model mistral:7b")
        print("  python run_nca_v1.py")
        sys.exit(1)

    # Load and analyze single-agent results
    single_stats = {}
    for name, path in SINGLE_FILES.items():
        records = load_jsonl(path)
        single_stats[name] = analyze_single(records)

    # Load and analyze NCA results
    nca_records = load_jsonl(NCA_FILE)
    nca = analyze_nca(nca_records)

    # Single-agent average error rate
    total_single_wrong = sum(s["n_wrong"] for s in single_stats.values())
    total_single_n = sum(s["n"] for s in single_stats.values())
    avg_single_error = total_single_wrong / total_single_n if total_single_n else 0

    # z-test: NCA error rate vs single-agent average
    # Using error counts: NCA n_wrong/n vs single avg n_wrong/n
    z_err, p_err = two_prop_ztest(
        total_single_wrong, total_single_n,
        nca["n_wrong"], nca["n"],
    )
    delta_err = (nca["error_rate"] - avg_single_error) * 100

    # ── Output ────────────────────────────────────────────────────────────

    print("=" * 64)
    print("MVE-20260404-01 Results")
    print("=" * 64)

    print()
    print("Single-agent error rates:")
    for name, s in single_stats.items():
        print(f"  {name:<14} {s['error_rate']*100:5.1f}% "
              f"[{s['ci_lo']*100:.1f}, {s['ci_hi']*100:.1f}] "
              f"({s['n_wrong']}/{s['n']})")
    print(f"  {'Average':<14} {avg_single_error*100:5.1f}% "
          f"({total_single_wrong}/{total_single_n})")

    print()
    print("NCA-LLM (3 nodes, steps=3):")
    print(f"  Error rate:    {nca['error_rate']*100:5.1f}% "
          f"[{nca['er_ci_lo']*100:.1f}, {nca['er_ci_hi']*100:.1f}] "
          f"({nca['n_wrong']}/{nca['n']})")
    print(f"  CFR:           {nca['cfr']*100:5.1f}% "
          f"[{nca['cfr_ci_lo']*100:.1f}, {nca['cfr_ci_hi']*100:.1f}] "
          f"({nca['n_wrong_unanimous']}/{nca['n_unanimous']})")
    print(f"  Unanimity:     {nca['unanimity_rate']*100:5.1f}% "
          f"({nca['n_unanimous']}/{nca['n']})")

    print()
    print("=" * 64)

    # Criterion 1: NCA error > single-agent average
    crit1 = nca["error_rate"] > avg_single_error
    print(f"Criterion 1: NCA error rate > single-agent average")
    print(f"  NCA={nca['error_rate']*100:.1f}% vs avg={avg_single_error*100:.1f}%")
    print(f"  delta: {delta_err:+.1f}pp  z={z_err:.4f}  p={p_err:.4f}")
    print(f"  -> {'SUCCESS' if crit1 else 'FAIL'}")

    print()

    # Criterion 2: CFR > 20%
    crit2 = nca["cfr"] > 0.20
    print(f"Criterion 2: CFR > 20%")
    print(f"  CFR = {nca['cfr']*100:.1f}%")
    print(f"  -> {'SUCCESS' if crit2 else 'FAIL'}")

    print()

    # Reference: unanimity > 50%
    ref_unan = nca["unanimity_rate"] > 0.50
    print(f"Reference: unanimity rate > 50%")
    print(f"  unanimity = {nca['unanimity_rate']*100:.1f}%")
    print(f"  -> {'SUCCESS' if ref_unan else 'FAIL'}")

    print()
    print("=" * 64)

    # Comparison with exploration phase
    print("Exploration phase reference (v2-v4 average):")
    print(f"  Old CFR average: 48.2%")
    print(f"  New CFR:         {nca['cfr']*100:.1f}%")

    print()
    print("=" * 64)

    # Overall verdict
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
    print("=" * 64)


if __name__ == "__main__":
    main()
