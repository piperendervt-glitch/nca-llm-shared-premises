"""
analyze_v8.py

MVE-20260404-08: Common sense reasoning vs world_consistency comparison.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_V8 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v8"
RESULTS_V1 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v1"
RESULTS_V2 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v2"


def load_jsonl(path):
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

    # Calibration error
    confs = []
    for r in rows:
        no = r.get("node_outputs", {})
        for role in ("solver", "verifier", "critic"):
            if role in no:
                c = no[role].get("confidence")
                if c is not None:
                    confs.append(float(c))
    mean_conf = np.mean(confs) if confs else 0
    cal_err = abs(mean_conf - n_correct / n_total) if n_total else 0

    return {
        "n_total": n_total, "n_unan": n_unan, "n_wrong_unan": n_wrong_unan,
        "cfr": cfr, "uni_rate": uni_rate,
        "n_split": n_split, "n_split_correct": n_split_correct,
        "split_acc": split_acc, "ovr_acc": ovr_acc,
        "cal_err": cal_err * 100, "split_rate": (n_split / n_total * 100) if n_total else 0,
    }


def main():
    cs_homo = load_jsonl(RESULTS_V8 / "common_sense_7b_homo.jsonl")
    cs_het = load_jsonl(RESULTS_V8 / "common_sense_7b_het.jsonl")
    wc_homo = load_jsonl(RESULTS_V2 / "nca_7b_results.jsonl")
    wc_het = load_jsonl(RESULTS_V1 / "nca_v1_results.jsonl")

    s_ch = stats(cs_homo)
    s_ce = stats(cs_het)
    s_wh = stats(wc_homo)
    s_we = stats(wc_het)

    print("=" * 80)
    print("MVE-20260404-08 Results: Common Sense Reasoning")
    print("=" * 80)
    print()

    hdr = f"{'Condition x Task':<22} | {'n_tot':>5} | {'n_uni':>5} | {'n_wu':>4} | {'CFR':>7} | {'uni%':>6} | {'n_spl':>5} | {'spl_acc':>7} | {'ovr_acc':>7}"
    sep = "-" * 22 + "-+-" + "-" * 5 + "-+-" + "-" * 5 + "-+-" + "-" * 4 + "-+-" + "-" * 7 + "-+-" + "-" * 6 + "-+-" + "-" * 5 + "-+-" + "-" * 7 + "-+-" + "-" * 7
    print(hdr)
    print(sep)

    for label, s in [("7b_homo x WC(ref)", s_wh), ("7b_het  x WC(ref)", s_we),
                      ("7b_homo x CS", s_ch), ("7b_het  x CS", s_ce)]:
        print(f"{label:<22} | {s['n_total']:>5} | {s['n_unan']:>5} | {s['n_wrong_unan']:>4} | {s['cfr']:>6.1f}% | {s['uni_rate']:>5.1f}% | {s['n_split']:>5} | {s['split_acc']:>6.1f}% | {s['ovr_acc']:>6.1f}%")

    print()
    print("=" * 80)
    print("Key comparison:")
    cs_cfr_diff = s_ch["cfr"] - s_ce["cfr"]
    wc_cfr_diff = s_wh["cfr"] - s_we["cfr"]
    print(f"  CFR diff (CS):  CFR(homo) - CFR(het) = {s_ch['cfr']:.1f}% - {s_ce['cfr']:.1f}% = {cs_cfr_diff:+.1f}pp")
    print(f"  CFR diff (WC):  CFR(homo) - CFR(het) = {s_wh['cfr']:.1f}% - {s_we['cfr']:.1f}% = {wc_cfr_diff:+.1f}pp (reference)")
    print()

    print("=" * 80)
    print("Success Criteria (pre-declared, DO NOT CHANGE):")
    must1 = cs_cfr_diff > 5
    must2 = cs_cfr_diff > 0
    print(f"  Must 1: CFR diff > +5pp:    {'SUCCESS' if must1 else 'FAIL'} ({cs_cfr_diff:+.1f}pp)")
    print(f"  Must 2: Positive direction:  {'SUCCESS' if must2 else 'FAIL'}")
    print()

    if must1 and must2:
        verdict = "SUCCESS"
    elif must1 or must2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"  Verdict: {verdict}")
    print()

    print("=" * 80)
    print("Task type breakdown (common sense):")
    print()
    task_types = sorted(set(r["task_type"] for r in cs_homo))
    print(f"{'type':<14} | {'homo CFR':>9} | {'het CFR':>9} | {'homo spl_acc':>12} | {'het spl_acc':>12}")
    print("-" * 14 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 12 + "-+-" + "-" * 12)
    for tt in task_types:
        homo_tt = [r for r in cs_homo if r["task_type"] == tt]
        het_tt = [r for r in cs_het if r["task_type"] == tt]
        sh = stats(homo_tt)
        se = stats(het_tt)
        cfr_h = f"{sh['cfr']:.1f}%" if sh['n_unan'] > 0 else "n/a"
        cfr_e = f"{se['cfr']:.1f}%" if se['n_unan'] > 0 else "n/a"
        sa_h = f"{sh['split_acc']:.1f}%" if sh['n_split'] > 0 else "n/a"
        sa_e = f"{se['split_acc']:.1f}%" if se['n_split'] > 0 else "n/a"
        print(f"{tt:<14} | {cfr_h:>9} | {cfr_e:>9} | {sa_h:>12} | {sa_e:>12}")

    print()
    print("=" * 80)
    print("A-4 refined connection:")
    print(f"  split_rate (7b_het x CS): {s_ce['split_rate']:.1f}%")
    print(f"  cal_error  (7b_het x CS): {s_ce['cal_err']:.1f}%")
    print(f"  -> H*_i divergence {'observed' if s_ce['split_rate'] > 30 else 'NOT clearly observed'}.")
    print("=" * 80)


if __name__ == "__main__":
    main()
