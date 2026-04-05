"""
analyze_v10.py

MVE-20260405-01: Compare external tasks (Grok/ChatGPT) with Claude WC.
"""

import json
from pathlib import Path

RESULTS_V10 = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v10"
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

    return {
        "n_total": n_total, "n_unan": n_unan, "n_wrong_unan": n_wrong_unan,
        "cfr": cfr, "uni_rate": uni_rate,
        "n_split": n_split, "n_split_correct": n_split_correct,
        "split_acc": split_acc, "ovr_acc": ovr_acc,
    }


def main():
    grok_homo = load_jsonl(RESULTS_V10 / "grok_7b_homo.jsonl")
    grok_het = load_jsonl(RESULTS_V10 / "grok_7b_het.jsonl")
    chatgpt_homo = load_jsonl(RESULTS_V10 / "chatgpt_7b_homo.jsonl")
    chatgpt_het = load_jsonl(RESULTS_V10 / "chatgpt_7b_het.jsonl")
    wc_homo = load_jsonl(RESULTS_V2 / "nca_7b_results.jsonl")
    wc_het = load_jsonl(RESULTS_V1 / "nca_v1_results.jsonl")

    s_gh = stats(grok_homo)
    s_ge = stats(grok_het)
    s_ch = stats(chatgpt_homo)
    s_ce = stats(chatgpt_het)
    s_wh = stats(wc_homo)
    s_we = stats(wc_het)

    print("=" * 80)
    print("MVE-20260405-01: External Task Verification Phase")
    print("=" * 80)
    print()

    hdr = f"{'Condition x Task':<24} | {'n_tot':>5} | {'n_uni':>5} | {'n_wu':>4} | {'CFR':>7} | {'uni%':>6} | {'n_spl':>5} | {'spl_acc':>7} | {'ovr_acc':>7}"
    sep = "-" * 24 + "-+-" + "-" * 5 + "-+-" + "-" * 5 + "-+-" + "-" * 4 + "-+-" + "-" * 7 + "-+-" + "-" * 6 + "-+-" + "-" * 5 + "-+-" + "-" * 7 + "-+-" + "-" * 7
    print(hdr)
    print(sep)

    for label, s in [("7b_homo x Claude WC", s_wh), ("7b_het  x Claude WC", s_we),
                      ("7b_homo x Grok", s_gh), ("7b_het  x Grok", s_ge),
                      ("7b_homo x ChatGPT", s_ch), ("7b_het  x ChatGPT", s_ce)]:
        print(f"{label:<24} | {s['n_total']:>5} | {s['n_unan']:>5} | {s['n_wrong_unan']:>4} | {s['cfr']:>6.1f}% | {s['uni_rate']:>5.1f}% | {s['n_split']:>5} | {s['split_acc']:>6.1f}% | {s['ovr_acc']:>6.1f}%")

    print()
    print("=" * 80)
    print("CFR diff comparison:")
    wc_diff = s_wh["cfr"] - s_we["cfr"]
    grok_diff = s_gh["cfr"] - s_ge["cfr"]
    chatgpt_diff = s_ch["cfr"] - s_ce["cfr"]
    print(f"  Claude WC:  {s_wh['cfr']:.1f}% - {s_we['cfr']:.1f}% = {wc_diff:+.1f}pp (reference)")
    print(f"  Grok:       {s_gh['cfr']:.1f}% - {s_ge['cfr']:.1f}% = {grok_diff:+.1f}pp")
    print(f"  ChatGPT:    {s_ch['cfr']:.1f}% - {s_ce['cfr']:.1f}% = {chatgpt_diff:+.1f}pp")

    print()
    print("=" * 80)
    print("Success Criteria (pre-declared, DO NOT CHANGE):")
    grok_ok = grok_diff > 5 and grok_diff > 0
    chatgpt_ok = chatgpt_diff > 5 and chatgpt_diff > 0
    print(f"  Set A (Grok)    CFR diff > +5pp: {'SUCCESS' if grok_ok else 'FAIL'} ({grok_diff:+.1f}pp)")
    print(f"  Set B (ChatGPT) CFR diff > +5pp: {'SUCCESS' if chatgpt_ok else 'FAIL'} ({chatgpt_diff:+.1f}pp)")

    if grok_ok and chatgpt_ok:
        verdict = "FULL SUCCESS"
    elif grok_ok or chatgpt_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print(f"\n  Verdict: {verdict}")
    if verdict == "FULL SUCCESS":
        print("  -> External validity ESTABLISHED")
        print("  -> Exploration phase findings are confirmed")
    elif verdict == "PARTIAL":
        print("  -> Partial confirmation - results vary by task source")
    else:
        print("  -> Exploration phase findings NOT confirmed externally")
        print("  -> May need to redesign verification approach")

    print("=" * 80)

    # Save summary
    summary = {
        "claude_wc": {"cfr_homo": round(s_wh["cfr"], 1), "cfr_het": round(s_we["cfr"], 1), "diff": round(wc_diff, 1)},
        "grok": {"cfr_homo": round(s_gh["cfr"], 1), "cfr_het": round(s_ge["cfr"], 1), "diff": round(grok_diff, 1)},
        "chatgpt": {"cfr_homo": round(s_ch["cfr"], 1), "cfr_het": round(s_ce["cfr"], 1), "diff": round(chatgpt_diff, 1)},
        "verdict": verdict,
    }
    out_path = RESULTS_V10 / "verification_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
