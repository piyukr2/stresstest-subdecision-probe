"""
Week 4 analysis: stress-induced logit shift vs. acoustic noise floor.

Reads the eight JSONL pathways produced in Weeks 2-3 and computes:

  Pathway summaries (sanity)
    Argmax accuracy and pair-consistency accuracy per pathway, per format.

  Headline 1 — decision-flip rate
    Within each sentence (same text, different stress patterns), what fraction
    of within-sentence pairs flip the model's argmax in the *correct* direction
    (i.e. each member of the pair lands on its own gold label)?
    Reported with bootstrap 95% CI over sentences.

  Headline 2 — stress shift vs. noise floor
    For each item i with a sister item j (same sentence, different stress)
    and each perturbation p, build paired observations:
        stress_shift_i = |p_first(i) - p_first(j)|   (averaged over sisters
                                                     for 3-stress sentences)
        noise_shift_i_p = |p_first(clean_i) - p_first(perturbed_i_p)|
    Paired Wilcoxon signed-rank tests whether stress_shift > noise_shift.
    The "strictest" perturbation is the one with the largest 95th-percentile
    noise-floor shift; the headline test is stress-shift vs. strictest-floor.

Pre-registered: two Wilcoxon tests total (one per answer format). Bootstrap
95% CIs on every reported quantity, since N is modest.

Inputs (read from --in-dir, default ./results):
    logits_primary_{A,B}.jsonl
    logits_textonly_{A,B}.jsonl                    (context only)
    logits_cascade_{A,B}.jsonl                      (context only)
    logits_noise_{gain,noise,stretch_slow,stretch_fast}_{A,B}.jsonl

Outputs:
    results/analysis_summary.json   (machine-readable)
    Console table (human-readable)

Run locally — pure CPU. No model loads.
    python scripts/analysis.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("scipy is required. pip install scipy")
    sys.exit(1)

FORMATS = ("A", "B")
PERTURBATIONS = ("gain", "noise", "stretch_slow", "stretch_fast")
PATHWAYS_CONTEXT = ("primary", "textonly", "cascade")
RNG = np.random.default_rng(seed=20260615)
BOOTSTRAP_ITERS = 10_000


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_idx(rows):
    return {r["idx"]: r for r in rows}


def p_label(rec):
    """Probability the model places on the gold-label option."""
    return rec["p_first"] if rec["label"] == 0 else rec["p_second"]


def accuracy(rows):
    n = len(rows)
    if not n:
        return float("nan")
    return sum(r["correct"] for r in rows) / n


def group_by_sentence(rows):
    """Group items that share a transcription (= same sentence, varying stress).
    Cascade records use `gold_transcription`; primary/text-only/noise records
    use `transcription`. Accept either."""
    groups = defaultdict(list)
    for r in rows:
        key = r.get("transcription") or r.get("gold_transcription")
        groups[key].append(r)
    return groups


def pair_consistency_accuracy(rows):
    """Fraction of within-sentence pairs where BOTH items are correct."""
    groups = group_by_sentence(rows)
    n_pairs, n_both_correct = 0, 0
    for items in groups.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                n_pairs += 1
                if items[i]["correct"] and items[j]["correct"]:
                    n_both_correct += 1
    return n_both_correct / n_pairs if n_pairs else float("nan"), n_pairs


def decision_flip_rate(rows):
    """
    For each within-sentence pair, did the model's argmax flip in the
    correct direction? "Correct direction" = each member lands on its own
    gold label. Only pairs whose two items have *different* gold labels
    are eligible (otherwise there's no flip to make).
    Returns (rate, n_eligible_pairs, per_pair_indicator_array).
    """
    groups = group_by_sentence(rows)
    indicators = []
    for items in groups.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["label"] == b["label"]:
                    continue
                hit = int(a["argmax"] == a["label"] and b["argmax"] == b["label"])
                indicators.append(hit)
    arr = np.array(indicators, dtype=np.float64)
    rate = float(arr.mean()) if arr.size else float("nan")
    return rate, arr.size, arr


def stress_shifts(rows):
    """
    Per-item stress shift: |p_first(item) - p_first(sister)|, averaged over
    sisters when a sentence has >2 stress patterns. Returns a dict idx -> shift.
    """
    groups = group_by_sentence(rows)
    out = {}
    for items in groups.values():
        if len(items) < 2:
            continue
        ps = {r["idx"]: r["p_first"] for r in items}
        for r in items:
            others = [v for k, v in ps.items() if k != r["idx"]]
            out[r["idx"]] = float(np.mean([abs(r["p_first"] - o) for o in others]))
    return out


def noise_shifts(primary, perturbed):
    """idx -> |p_first(clean) - p_first(perturbed)| over indices in both."""
    out = {}
    for idx, p in primary.items():
        q = perturbed.get(idx)
        if q is None:
            continue
        out[idx] = float(abs(p["p_first"] - q["p_first"]))
    return out


def bootstrap_ci(arr, fn=np.mean, iters=BOOTSTRAP_ITERS, alpha=0.05):
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    idx = RNG.integers(0, n, size=(iters, n))
    stats = fn(arr[idx], axis=1)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def percentile(arr, q):
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(np.asarray(arr), q))


def analyze_format(in_dir, fmt):
    print(f"\n{'=' * 72}\nFORMAT {fmt}\n{'=' * 72}")

    # Load every available pathway. Missing files are tolerated (with a warning).
    def maybe_read(name):
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            print(f"  [warn] missing: {path}")
            return None
        return read_jsonl(path)

    primary_rows = maybe_read(f"logits_primary_{fmt}.jsonl")
    if primary_rows is None:
        print(f"  [skip] no primary pathway for format {fmt}")
        return None

    context_rows = {"primary": primary_rows}
    for pw, fname in (
        ("textonly", f"logits_textonly_{fmt}.jsonl"),
        ("cascade", f"logits_cascade_{fmt}.jsonl"),
        ("text_rescue", f"logits_text_rescue_{fmt}.jsonl"),
    ):
        rows = maybe_read(fname)
        if rows is not None:
            context_rows[pw] = rows

    noise_rows = {}
    for p in PERTURBATIONS:
        rows = maybe_read(f"logits_noise_{p}_{fmt}.jsonl")
        if rows is not None:
            noise_rows[p] = rows

    # ---------- Pathway accuracies ----------
    print("\nPathway accuracies (argmax)  /  pair-consistency  / N pairs")
    pathway_summary = {}
    for name, rows in context_rows.items():
        acc = accuracy(rows)
        pc, npairs = pair_consistency_accuracy(rows)
        print(f"  {name:>10}  acc={acc:.3f}   pair-both-correct={pc:.3f}  (N={npairs})")
        pathway_summary[name] = {
            "accuracy": acc, "pair_both_correct": pc, "n_pairs": npairs
        }
    for p, rows in noise_rows.items():
        acc = accuracy(rows)
        print(f"  noise/{p:>13}  acc={acc:.3f}")
        pathway_summary[f"noise_{p}"] = {"accuracy": acc}

    # ---------- Headline 1: decision-flip rate (primary) ----------
    flip_rate, n_eligible, flip_arr = decision_flip_rate(primary_rows)
    flip_ci = bootstrap_ci(flip_arr)
    print(f"\nHeadline 1 — decision-flip rate (primary, pairs w/ different labels):")
    print(f"  rate = {flip_rate:.3f}   95% CI = ({flip_ci[0]:.3f}, {flip_ci[1]:.3f})   "
          f"N eligible pairs = {n_eligible}")

    # ---------- Headline 2: stress shift vs. noise floor ----------
    primary_by_idx = by_idx(primary_rows)
    stress_by_idx = stress_shifts(primary_rows)
    stress_vals = np.array(list(stress_by_idx.values()))
    print(f"\nStress-shift distribution (primary, |Δp_first| vs sister(s)):")
    print(f"  N items = {len(stress_vals)}   "
          f"median={percentile(stress_vals, 0.5):.4f}   "
          f"mean={float(stress_vals.mean()):.4f}   "
          f"95th pctile={percentile(stress_vals, 0.95):.4f}")
    ss_lo, ss_hi = bootstrap_ci(stress_vals, fn=np.median)
    print(f"  bootstrap 95% CI on median: ({ss_lo:.4f}, {ss_hi:.4f})")

    print("\nNoise-floor shift distributions (per perturbation, |Δp_first|):")
    floor_distributions = {}
    for p, rows in noise_rows.items():
        ns = noise_shifts(primary_by_idx, by_idx(rows))
        vals = np.array(list(ns.values()))
        floor_distributions[p] = ns
        p95 = percentile(vals, 0.95)
        print(f"  {p:>13}  N={len(vals):>4}  "
              f"median={percentile(vals, 0.5):.4f}  "
              f"mean={float(vals.mean()):.4f}  "
              f"95th={p95:.4f}")

    if not floor_distributions:
        print("  [skip] no perturbations available; cannot run headline test.")
        return {
            "format": fmt,
            "pathway_summary": pathway_summary,
            "decision_flip_rate": flip_rate,
            "decision_flip_ci": flip_ci,
            "n_eligible_pairs": n_eligible,
            "stress_shift_median": percentile(stress_vals, 0.5),
            "stress_shift_median_ci": (ss_lo, ss_hi),
        }

    # Strictest = perturbation with the largest 95th-percentile shift.
    strictest = max(
        floor_distributions,
        key=lambda p: percentile(list(floor_distributions[p].values()), 0.95),
    )
    strictest_vals = np.array(list(floor_distributions[strictest].values()))
    strictest_p95 = percentile(strictest_vals, 0.95)
    print(f"\nStrictest perturbation = {strictest}  (95th pctile = {strictest_p95:.4f})")

    # Paired Wilcoxon: stress_shift vs strictest noise-floor shift on shared idx.
    shared_idx = sorted(set(stress_by_idx) & set(floor_distributions[strictest]))
    paired_stress = np.array([stress_by_idx[i] for i in shared_idx])
    paired_noise = np.array([floor_distributions[strictest][i] for i in shared_idx])
    diffs = paired_stress - paired_noise

    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_zero = int((diffs == 0).sum())
    print(f"\nHeadline 2 — paired Wilcoxon (stress_shift vs {strictest} floor):")
    print(f"  N paired items = {len(diffs)}   "
          f"+ {n_pos}   - {n_neg}   = {n_zero}")
    print(f"  median Δ = {percentile(diffs, 0.5):.4f}   "
          f"mean Δ = {float(diffs.mean()):.4f}")
    md_lo, md_hi = bootstrap_ci(diffs, fn=np.median)
    print(f"  bootstrap 95% CI on median Δ: ({md_lo:.4f}, {md_hi:.4f})")

    try:
        wstat, wp = wilcoxon(paired_stress, paired_noise, alternative="greater")
        print(f"  Wilcoxon (one-sided, stress > floor): W={wstat:.1f}  p={wp:.4g}")
    except ValueError as e:
        wstat, wp = float("nan"), float("nan")
        print(f"  Wilcoxon failed: {e}")

    pre_reg_pass = bool(percentile(stress_vals, 0.5) > strictest_p95)
    print(
        f"\nPre-registered check: stress-shift median ({percentile(stress_vals, 0.5):.4f}) "
        f"> 95th pctile of strictest floor ({strictest_p95:.4f})?  "
        f"{'PASS' if pre_reg_pass else 'FAIL'}"
    )

    return {
        "format": fmt,
        "pathway_summary": pathway_summary,
        "decision_flip_rate": flip_rate,
        "decision_flip_ci": flip_ci,
        "n_eligible_pairs": n_eligible,
        "stress_shift_median": percentile(stress_vals, 0.5),
        "stress_shift_mean": float(stress_vals.mean()),
        "stress_shift_p95": percentile(stress_vals, 0.95),
        "stress_shift_median_ci": (ss_lo, ss_hi),
        "strictest_perturbation": strictest,
        "strictest_p95": strictest_p95,
        "wilcoxon_W": float(wstat),
        "wilcoxon_p": float(wp),
        "paired_diff_median": percentile(diffs, 0.5),
        "paired_diff_median_ci": (md_lo, md_hi),
        "pre_registered_pass": pre_reg_pass,
        "n_paired_items": len(diffs),
    }


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_in = os.path.join(project_root, "results")
    default_out = os.path.join(project_root, "results", "analysis_summary.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=default_in)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    summary = {}
    for fmt in FORMATS:
        result = analyze_format(args.in_dir, fmt)
        if result is not None:
            summary[fmt] = result

    print(f"\n{'=' * 72}\nFINAL HEADLINES\n{'=' * 72}")
    for fmt, s in summary.items():
        ci = s.get("decision_flip_ci", (float("nan"), float("nan")))
        print(f"  Format {fmt}:")
        print(
            f"    Decision-flip rate = {s['decision_flip_rate']:.3f}  "
            f"95% CI ({ci[0]:.3f}, {ci[1]:.3f})  "
            f"on N={s['n_eligible_pairs']} pairs"
        )
        if "wilcoxon_p" in s:
            print(
                f"    Stress shift vs strictest floor ({s['strictest_perturbation']}): "
                f"Wilcoxon p={s['wilcoxon_p']:.4g}, "
                f"median Δ={s['paired_diff_median']:.4f} "
                f"95% CI ({s['paired_diff_median_ci'][0]:.4f}, "
                f"{s['paired_diff_median_ci'][1]:.4f})  "
                f"pre-reg: {'PASS' if s['pre_registered_pass'] else 'FAIL'}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
