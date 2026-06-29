"""
Week 5 figures. Reads the JSONLs in results/ and writes PNGs to figures/.

Three figures, each with a Format-A panel and a Format-B panel:

  figures/fig1_shift_distributions.png
    Stress-shift distribution overlaid with the four noise-floor shift
    distributions. Vertical line at the strictest perturbation's 95th
    percentile (this is the pre-registered threshold that the headline
    test failed). Makes the Wilcoxon result visible.

  figures/fig2_stress_vs_floor_scatter.png
    Per-item scatter: x = stress shift, y = strictest-floor shift.
    Diagonal y=x. Points colored by whether the primary model got that
    item correct. Points above the diagonal are floor > stress (bad);
    points below are stress > floor (good).

  figures/fig3_flip_by_position.png
    Decision-flip rate broken down by sentence-stress position
    (initial / medial / final), with bootstrap 95% CIs. Position is
    the average word-fraction position of the two members' stressed
    words. Pairs with same gold label are excluded (no flip to make).

Run locally — pure CPU, no model loads.
    python scripts/figures.py
"""

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. pip install matplotlib")
    sys.exit(1)


FORMATS = ("A", "B")
PERTURBATIONS = ("gain", "noise", "stretch_slow", "stretch_fast")
PERTURBATION_LABELS = {
    "gain": "gain (−3 dB)",
    "noise": "noise (~30 dB SNR)",
    "stretch_slow": "stretch ×0.98",
    "stretch_fast": "stretch ×1.02",
}
RNG = np.random.default_rng(seed=20260629)
BOOTSTRAP_ITERS = 10_000
WORD_RE = re.compile(r"\b[\w'-]+\b")


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


def group_by_sentence(rows):
    groups = defaultdict(list)
    for r in rows:
        key = r.get("transcription") or r.get("gold_transcription")
        groups[key].append(r)
    return groups


def stress_shifts(rows):
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


def noise_shifts(primary_by_idx, perturbed_by_idx):
    out = {}
    for idx, p in primary_by_idx.items():
        q = perturbed_by_idx.get(idx)
        if q is None:
            continue
        out[idx] = float(abs(p["p_first"] - q["p_first"]))
    return out


def strictest_perturbation(floor_distributions):
    return max(
        floor_distributions,
        key=lambda p: float(np.quantile(list(floor_distributions[p].values()), 0.95)),
    )


def bootstrap_ci(arr, fn=np.mean, iters=BOOTSTRAP_ITERS, alpha=0.05):
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    idx = RNG.integers(0, n, size=(iters, n))
    stats = fn(arr[idx], axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


# ---------- stress-position helpers ----------

def stress_position_fraction(item):
    """Return the average word-fraction position of the stressed words
    in this item, in [0, 1]. None if we can't determine it."""
    sp = item.get("stress_pattern")
    transcription = item.get("transcription") or item.get("gold_transcription") or ""
    n_words = len(WORD_RE.findall(transcription))
    if n_words == 0:
        return None
    if isinstance(sp, dict):
        idxs = sp.get("indices") or []
        if not idxs:
            binary = sp.get("binary") or []
            idxs = [i for i, b in enumerate(binary) if int(b) == 1]
    else:
        return None
    if not idxs:
        return None
    mean_idx = sum(int(i) for i in idxs) / len(idxs)
    return mean_idx / max(1, n_words - 1) if n_words > 1 else 0.5


def bin_position(frac):
    if frac is None:
        return None
    if frac < 1 / 3:
        return "initial"
    if frac < 2 / 3:
        return "medial"
    return "final"


# ---------- figure 1: shift distributions ----------

def fig1_shift_distributions(in_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, fmt in zip(axes, FORMATS):
        primary = read_jsonl(os.path.join(in_dir, f"logits_primary_{fmt}.jsonl"))
        primary_idx = by_idx(primary)
        stress = np.array(list(stress_shifts(primary).values()))

        floors = {}
        for p in PERTURBATIONS:
            path = os.path.join(in_dir, f"logits_noise_{p}_{fmt}.jsonl")
            if not os.path.exists(path):
                continue
            floors[p] = noise_shifts(primary_idx, by_idx(read_jsonl(path)))

        strict = strictest_perturbation(floors)
        strict_p95 = float(np.quantile(list(floors[strict].values()), 0.95))

        bins = np.linspace(0, 1, 31)
        ax.hist(stress, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"stress (median={np.median(stress):.3f})", density=True)
        floor_colors = {
            "gain": "#bbbbbb",
            "noise": "#888888",
            "stretch_slow": "#d62728",
            "stretch_fast": "#ff7f0e",
        }
        for p in PERTURBATIONS:
            if p not in floors:
                continue
            vals = np.array(list(floors[p].values()))
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.5,
                    color=floor_colors[p],
                    label=f"{PERTURBATION_LABELS[p]} (95th={np.quantile(vals, 0.95):.3f})",
                    density=True)

        ax.axvline(strict_p95, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"strictest 95th = {strict_p95:.3f}")
        ax.set_title(f"Format {fmt} — stress shift vs noise-floor shifts")
        ax.set_xlabel("|Δ p_first|")
        if fmt == "A":
            ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- figure 2: stress vs strictest-floor scatter ----------

def fig2_scatter(in_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax, fmt in zip(axes, FORMATS):
        primary = read_jsonl(os.path.join(in_dir, f"logits_primary_{fmt}.jsonl"))
        primary_idx = by_idx(primary)
        stress = stress_shifts(primary)

        floors = {}
        for p in PERTURBATIONS:
            path = os.path.join(in_dir, f"logits_noise_{p}_{fmt}.jsonl")
            if os.path.exists(path):
                floors[p] = noise_shifts(primary_idx, by_idx(read_jsonl(path)))
        strict = strictest_perturbation(floors)
        floor = floors[strict]

        shared = sorted(set(stress) & set(floor))
        xs = np.array([stress[i] for i in shared])
        ys = np.array([floor[i] for i in shared])
        correct = np.array([primary_idx[i]["correct"] for i in shared], dtype=bool)

        ax.scatter(xs[correct], ys[correct], s=18, alpha=0.65, color="#2ca02c",
                   label=f"correct (n={int(correct.sum())})")
        ax.scatter(xs[~correct], ys[~correct], s=18, alpha=0.65, color="#d62728",
                   label=f"wrong (n={int((~correct).sum())})")
        lim = float(max(xs.max(), ys.max()) * 1.05)
        ax.plot([0, lim], [0, lim], color="#444444", linestyle="--",
                linewidth=1.0, label="y = x")

        n_below = int((xs > ys).sum())
        ax.set_title(
            f"Format {fmt} (strictest = {strict})\n"
            f"stress > floor: {n_below}/{len(xs)} ({100*n_below/len(xs):.0f}%)"
        )
        ax.set_xlabel("stress shift")
        if fmt == "A":
            ax.set_ylabel("strictest noise-floor shift")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- figure 3: decision-flip rate by stress position ----------

def fig3_flip_by_position(in_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    bins = ("initial", "medial", "final")
    x_pos = np.arange(len(bins))

    for ax, fmt in zip(axes, FORMATS):
        primary = read_jsonl(os.path.join(in_dir, f"logits_primary_{fmt}.jsonl"))
        groups = group_by_sentence(primary)

        per_bin = defaultdict(list)  # bin -> list of {0,1} flip indicators
        skipped = 0
        for items in groups.values():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if a["label"] == b["label"]:
                        continue
                    fa = stress_position_fraction(a)
                    fb = stress_position_fraction(b)
                    if fa is None or fb is None:
                        skipped += 1
                        continue
                    bucket = bin_position((fa + fb) / 2)
                    hit = int(a["argmax"] == a["label"] and b["argmax"] == b["label"])
                    per_bin[bucket].append(hit)

        heights, los, his, counts = [], [], [], []
        for b in bins:
            arr = np.array(per_bin.get(b, []), dtype=np.float64)
            counts.append(len(arr))
            if len(arr) == 0:
                heights.append(0.0)
                los.append(0.0)
                his.append(0.0)
                continue
            heights.append(float(arr.mean()))
            lo, hi = bootstrap_ci(arr, fn=np.mean)
            los.append(lo)
            his.append(hi)

        heights = np.array(heights)
        err_low = heights - np.array(los)
        err_high = np.array(his) - heights
        ax.bar(x_pos, heights, color="#1f77b4", alpha=0.75, edgecolor="#0b3a66",
               width=0.55,
               yerr=[err_low, err_high], capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": "#222"})

        for i, (h, c) in enumerate(zip(heights, counts)):
            ax.text(i, max(h, 0.01) + 0.012, f"n={c}", ha="center",
                    va="bottom", fontsize=9, color="#444")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bins)
        ax.set_ylim(0, max(0.35, max(his) + 0.05))
        ax.set_title(f"Format {fmt} — decision-flip rate by stress position")
        if fmt == "A":
            ax.set_ylabel("flip rate (both items correct)")
        ax.set_xlabel("avg stress position within sentence")
        if skipped:
            ax.text(0.02, 0.96, f"{skipped} pairs skipped (no stress_pattern)",
                    transform=ax.transAxes, fontsize=8, va="top", color="#777")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_dir = os.path.join(project_root, "results")
    out_dir = os.path.join(project_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    fig1_shift_distributions(in_dir, os.path.join(out_dir, "fig1_shift_distributions.png"))
    fig2_scatter(in_dir, os.path.join(out_dir, "fig2_stress_vs_floor_scatter.png"))
    fig3_flip_by_position(in_dir, os.path.join(out_dir, "fig3_flip_by_position.png"))


if __name__ == "__main__":
    main()
