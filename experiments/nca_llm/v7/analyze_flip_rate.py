"""
analyze_flip_rate.py

MVE-20260404-07: Correlate flip rate with split rate, calibration error, and CFR diff.
Pre-declared thresholds: r < 0.5 for independence.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

RESULTS_V7 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v7"
CAL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "calibration_analysis.json"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    # Load flip rate results
    flip_files = {
        "world_consist": RESULTS_V7 / "flip_wc.jsonl",
        "logic_syllogism": RESULTS_V7 / "flip_syllogism.jsonl",
        "logic_contradict": RESULTS_V7 / "flip_contradiction.jsonl",
        "math_middle": RESULTS_V7 / "flip_math_mid.jsonl",
    }

    flip_rates = {}
    for name, path in flip_files.items():
        rows = load_jsonl(path)
        fr = np.mean([r["flip_rate"] for r in rows])
        flip_rates[name] = fr

    # Load existing calibration data
    with open(CAL_PATH, "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    # Reference metrics (7b_homo condition)
    # For logic subtypes, compute from v6 data
    ref = {
        "world_consist": {
            "split_rate": cal_data["conditions"]["7b_homo x WC"]["split_rate"],
            "cal_err": cal_data["conditions"]["7b_homo x WC"]["calibration_error"],
            "cfr_homo": cal_data["conditions"]["7b_homo x WC"]["cfr"],
            "cfr_het": cal_data["conditions"]["7b_het  x WC"]["cfr"],
        },
        "math_middle": {
            "split_rate": cal_data["conditions"]["7b_homo x math_mid"]["split_rate"],
            "cal_err": cal_data["conditions"]["7b_homo x math_mid"]["calibration_error"],
            "cfr_homo": cal_data["conditions"]["7b_homo x math_mid"]["cfr"],
            "cfr_het": cal_data["conditions"]["7b_het  x math_mid"]["cfr"],
        },
    }

    # Compute split_rate and cal_err for logic subtypes from v6 raw data
    v6_path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v6"
    for cond, task_type in [("logic_syllogism", "syllogism"), ("logic_contradict", "contradiction")]:
        homo_rows = [r for r in load_jsonl(v6_path / "logic_7b_homo.jsonl") if r["task_type"] == task_type]
        het_rows = [r for r in load_jsonl(v6_path / "logic_7b_het.jsonl") if r["task_type"] == task_type]

        n = len(homo_rows)
        n_unan_h = sum(1 for r in homo_rows if r["is_unanimous"])
        n_wu_h = sum(1 for r in homo_rows if r["is_unanimous"] and not r["is_correct"])
        n_split_h = n - n_unan_h
        n_correct_h = sum(1 for r in homo_rows if r["is_correct"])

        n_het = len(het_rows)
        n_unan_e = sum(1 for r in het_rows if r["is_unanimous"])
        n_wu_e = sum(1 for r in het_rows if r["is_unanimous"] and not r["is_correct"])

        # Confidence
        confs_h = []
        for r in homo_rows:
            for role in ("solver", "verifier", "critic"):
                if role in r.get("node_outputs", {}):
                    c = r["node_outputs"][role].get("confidence")
                    if c is not None:
                        confs_h.append(float(c))

        acc_h = n_correct_h / n if n else 0
        mean_conf_h = np.mean(confs_h) if confs_h else 0

        ref[cond] = {
            "split_rate": n_split_h / n if n else 0,
            "cal_err": abs(mean_conf_h - acc_h),
            "cfr_homo": n_wu_h / n_unan_h if n_unan_h else 0,
            "cfr_het": n_wu_e / n_unan_e if n_unan_e else 0,
        }

    # Build vectors for correlation
    task_order = ["world_consist", "logic_syllogism", "logic_contradict", "math_middle"]
    fr_vec = [flip_rates[t] for t in task_order]
    sr_vec = [ref[t]["split_rate"] for t in task_order]
    ce_vec = [ref[t]["cal_err"] for t in task_order]
    cfr_diff_vec = [ref[t]["cfr_homo"] - ref[t]["cfr_het"] for t in task_order]

    print("=" * 64)
    print("MVE-20260404-07: flip rate independence verification")
    print("=" * 64)
    print()

    hdr = f"{'Task set':<20} | {'flip_rate':>9} | {'split_rate':>10} | {'cal_err':>7} | {'CFR diff':>8}"
    print(hdr)
    print("-" * 20 + "-+-" + "-" * 9 + "-+-" + "-" * 10 + "-+-" + "-" * 7 + "-+-" + "-" * 8)
    for t in task_order:
        cd = ref[t]["cfr_homo"] - ref[t]["cfr_het"]
        print(f"{t:<20} | {flip_rates[t]*100:>8.1f}% | {ref[t]['split_rate']*100:>9.1f}% | {ref[t]['cal_err']*100:>6.1f}% | {cd*100:>+7.1f}pp")

    print()
    print("=" * 64)
    print("Correlation analysis (n=4 task sets):")
    print()

    pairs = [
        ("flip_rate vs split_rate", fr_vec, sr_vec, 0.5),
        ("flip_rate vs cal_error", fr_vec, ce_vec, 0.5),
        ("flip_rate vs CFR_diff", fr_vec, cfr_diff_vec, None),
    ]

    results = {}
    for name, v1, v2, threshold in pairs:
        r, p = pearsonr(v1, v2)
        results[name] = {"r": r, "p": p}
        if threshold is not None:
            status = "SUCCESS" if abs(r) < threshold else "FAIL"
            print(f"  {name}: r = {r:+.3f} (p = {p:.4f}) → {status} (threshold |r| < {threshold})")
        else:
            direction = "positive" if r > 0 else "negative"
            print(f"  {name}: r = {r:+.3f} (p = {p:.4f}) → {direction} direction")

    print()
    print("=" * 64)

    r1 = abs(results["flip_rate vs split_rate"]["r"]) < 0.5
    r2 = abs(results["flip_rate vs cal_error"]["r"]) < 0.5

    if r1 and r2:
        verdict = "SUCCESS"
    elif r1 or r2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print(f"Claim verdict: {verdict}")
    print(f"  Required 1 (flip vs split |r| < 0.5):    {'SUCCESS' if r1 else 'FAIL'}")
    print(f"  Required 2 (flip vs cal_error |r| < 0.5): {'SUCCESS' if r2 else 'FAIL'}")
    print()
    print("=" * 64)
    print("Interpretation:")
    print("  [Fill in after reviewing results]")
    print("=" * 64)

    # Save
    save_data = {
        "flip_rates": {k: round(v, 4) for k, v in flip_rates.items()},
        "reference_metrics": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ref.items()},
        "correlations": {k: {"r": round(v["r"], 4), "p": round(v["p"], 4)} for k, v in results.items()},
        "verdict": verdict,
    }
    out_path = RESULTS_V7 / "flip_rate_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
