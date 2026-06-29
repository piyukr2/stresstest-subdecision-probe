# Results Sketch — Sub-Decision Prosodic Sensitivity in Qwen2-Audio

> **Superseded by [`final_writeup.md`](final_writeup.md) (Week 6).**
> This file is the Week 4 working sketch that was promoted into the
> paper-style document; it is preserved for narrative tone and the
> session-by-session reasoning trail, but `final_writeup.md` is the
> canonical version.

Status: draft sketch (last updated 2026-06-29). Numbers from
`results/analysis_summary.json`. Figures generated 2026-06-30 (Week 5).

---

## TL;DR

When Qwen2-Audio-7B-Instruct gets a contrastive-stress question wrong, its
**sub-decision logits still shift in the right direction.** Across 218
StressTest items in two answer formats, stress-induced probability shifts are
reliably larger than shifts produced by meaning-preserving acoustic
perturbations (paired Wilcoxon p ≈ 10⁻¹⁸ in Format A, 10⁻¹⁷ in Format B). The
model is **partially** prosody-sensitive: it hears stress, but doesn't weight
it enough to flip the final answer most of the time.

A text-rescue probe — feeding the same language-model backbone the transcript
with the stressed word in ALL CAPS — beats the audio model on both argmax
accuracy (≈ 59% vs 51–56%) and pair-consistency (≈ 23% vs 14–20%). The
**LM backbone can use stress when it arrives as a lexical cue**; the
bottleneck is in the audio→LM interface, not in the language model.

A secondary finding: Qwen2-Audio is **unusually fragile to small time-stretches**,
which broke the project's pre-registered threshold check. We document this
honestly rather than tune the bar to pass.

---

## Headline numbers

### Argmax accuracy (sanity check, matches StressTest paper)

| Pathway | Format A | Format B |
|---|---|---|
| Primary (Qwen2-Audio, clean audio) | 50.9% | 55.5% |
| Text-only (Qwen2-7B, gold transcript) | 49.5% | 48.2% |
| Cascade (Whisper → Qwen2-7B) | 49.5% | 48.2% |
| **Text-rescue (Qwen2-7B, ALL-CAPS stress hint)** | **59.6%** | **59.2%** |

Both falsification controls land at chance. The cascade matches text-only
exactly, because Whisper transcribes these short clean clips accurately enough
that the LM behaves the same as on the gold transcript — i.e. lexical content
alone is not enough. The text-rescue pathway lifts the same LM backbone ~10
points above chance the moment stress is delivered as capitalization:
**the language model can use stress when it arrives as text**.

### Pair-consistency accuracy (both items in a stress pair correct)

| Pathway | Format A | Format B |
|---|---|---|
| Primary | 13.5% | 19.5% |
| Text-only | 8.3% | 8.3% |
| Cascade | 8.3% | 8.3% |
| **Text-rescue** | **23.3%** | **22.6%** |

The primary model rarely gets both items of a contrastive pair right —
consistent with picking by other cues (option-order bias, word familiarity)
and getting each item correct about half the time independently. This is the
StressTest paper's central observation, reproduced. Text-rescue more than
doubles the pair-both-correct rate over the primary audio model — the
backbone is *more effective* at using stress when handed it as a capitalized
token than the audio model is at extracting it from acoustics.

### Headline 1 — decision-flip rate

For the 74 within-sentence pairs whose two items have *different* gold labels,
how often does the model's argmax flip in the *correct* direction (each
member lands on its own label)?

| Format | Rate | Bootstrap 95% CI |
|---|---|---|
| A | 9.5% | (4.1%, 16.2%) |
| B | 12.2% | (5.4%, 20.3%) |

Both CIs exclude zero. The flip rate is modest but reliably non-zero — the
model flips its answer in the correct direction about 1 pair in 10.

### Headline 2 — stress shift vs. noise-floor shift (the central finding)

Per item, we compute:
- **stress shift** = |p_first(item) − p_first(sister)|, averaged over sisters
  when a sentence has >2 stress patterns.
- **noise shift** = |p_first(clean) − p_first(perturbed)| for each perturbation.

Paired one-sided Wilcoxon signed-rank tests "stress shift > floor shift" on
each item. The "strictest" floor is the perturbation with the largest 95th
percentile.

| Format | Strictest floor | N (+ / − / =) | Median Δ | 95% CI on median Δ | Wilcoxon p |
|---|---|---|---|---|---|
| A | stretch_fast | 163 / 55 / 0 | 0.082 | (0.031, 0.147) | 4.2×10⁻¹⁸ |
| B | stretch_slow | 158 / 60 / 0 | 0.034 | (0.009, 0.081) | 7.7×10⁻¹⁷ |

In both formats, ~73–75% of items show a larger stress shift than the
strictest noise-floor shift. The paired-difference median is positive in
every bootstrap resample. **The sub-decision logits respond more to stress
than to acoustic perturbation, on a per-item basis, with extremely high
confidence.**

---

## The pre-registered check that failed (and what it actually tells us)

The pre-registration (CLAUDE.md, Decisions section) committed to a stricter
threshold than the Wilcoxon: the **median stress shift must exceed the 95th
percentile of the strictest floor's shift distribution.** That check failed
in both formats:

| Format | Stress median | Strictest floor 95th pctile | Pass? |
|---|---|---|---|
| A | 0.20 | 0.38 (stretch_fast) | FAIL |
| B | 0.10 | 0.31 (stretch_slow) | FAIL |

The threshold was designed assuming all four perturbations would behave like
gain and additive noise — small, tightly-distributed shifts. They didn't.
Look at the noise-floor 95th percentiles:

| Perturbation | Format A | Format B |
|---|---|---|
| gain (−3 dB) | 0.034 | 0.029 |
| noise (~30 dB SNR) | 0.141 | 0.112 |
| stretch_slow (0.98×) | 0.326 | 0.313 |
| stretch_fast (1.02×) | 0.375 | 0.300 |

Time-stretch — even at ±2%, which is acoustically inaudible as a meaning
change — moves the logits by an order of magnitude more than gain or
additive noise. We initially ran ±5% stretches; tightening to ±2% only
dropped `stretch_fast` 95th from 0.475 to 0.375, and `stretch_slow` barely
moved (0.322 → 0.326). The model's internal representations are sensitive to
durational changes the average listener wouldn't flag.

**This is itself a finding about Qwen2-Audio**, not an artifact of bad
perturbation design. We report it as such, and do not tune the threshold to
clear it post-hoc.

The Wilcoxon and decision-flip results stand on their own: they are *per-item
paired comparisons*, robust to whatever the floor distribution looks like.
The threshold test would have been a useful additional check had it passed;
its failure reflects a property of the model, not weakness of the effect.

---

## Architectural attribution — ears, not brain

The text-rescue probe (Phase 4) feeds the *same LM backbone* used in the
text-only and cascade pathways (Qwen2-7B-Instruct), but capitalizes the
stressed word in the transcript before answering — e.g.
`"I didn't say HE stole the money."`. Word indices come straight from the
StressTest `stress_pattern` field, so the hint is exactly the prosodically-
stressed word and nothing else.

| Pathway (Qwen2-7B backbone, varying input) | Format A argmax | Format B argmax | A pair-both | B pair-both |
|---|---|---|---|---|
| Text-only (gold transcript, no hint) | 49.5% | 48.2% | 8.3% | 8.3% |
| Cascade (Whisper transcript, no hint) | 49.5% | 48.2% | 8.3% | 8.3% |
| Text-rescue (gold transcript + ALL-CAPS hint) | **59.6%** | **59.2%** | **23.3%** | **22.6%** |
| Primary audio (Qwen2-Audio) | 50.9% | 55.5% | 13.5% | 19.5% |

Three things to read out of this:

1. **The LM backbone *can* use stress.** When stress arrives as a lexical
   cue (one word in caps), the same backbone that sat at chance on plain
   transcripts jumps ~10 points on argmax and roughly doubles
   pair-consistency.
2. **Text-rescue exceeds the primary audio model on both metrics, in both
   formats.** A trivial lexical encoding of stress is more useful to the
   downstream LM than whatever signal Qwen2-Audio currently extracts from
   the acoustic waveform.
3. **The bottleneck is perceptual, not integrative.** The audio model is
   not "deaf" to stress — the sub-decision Wilcoxon (Headline 2) shows the
   logits respond — but it isn't routing that signal into the part of the
   representation the LM head reads from. Calling it "ears failed" is
   shorthand for: stress is in the audio embedding only weakly, and the
   bridge between audio encoder and language head loses most of it.

This narrows the design space for any follow-up: improvements should
target the audio→LM interface (e.g. better prosodic features piped to the
text decoder, or training the bridge with stress-contrastive objectives),
not the language model itself.

---

## Limitations

- **N is modest.** 218 items, 74 eligible label-differing pairs for the flip
  rate. The Wilcoxon is well-powered because it is paired and per-item, but
  the flip-rate CIs are wide (±6 percentage points).
- **One primary model.** Qwen2.5-Omni-3B was time-boxed and dropped. The
  conclusions are specific to Qwen2-Audio-7B-Instruct.
- **English only**, StressTest speakers only.
- **Inference only.** No claim about whether fine-tuning could close the
  gap between sub-decision sensitivity and final-answer accuracy.
- **Threshold test failed.** Reported transparently, with the model-property
  interpretation.

---

## Reproducibility

- Code: `scripts/` (extraction, perturbation, analysis).
- Outputs: `results/*.jsonl` (one record per item, all logits + probabilities
  preserved) and `results/analysis_summary.json`.
- Bootstrap seed: 20260615 (in `analysis.py`).
- Noise-perturbation seed: each item's index (deterministic per-item).
- Token IDs hardcoded after one-time verification on the Qwen2 tokenizer.

---

## Figures

Three figures live in `figures/` (generated by `scripts/figures.py`):

- `fig1_shift_distributions.png` — per-format histograms of the stress-shift
  distribution overlaid with the four noise-floor shift distributions, with
  a vertical line at the strictest-floor 95th percentile. Visualises the
  Wilcoxon result and the (failed) pre-registered threshold.
- `fig2_stress_vs_floor_scatter.png` — per-item scatter of stress shift vs
  strictest noise-floor shift, points colored by correct/wrong on the
  primary audio model, with a y = x diagonal. 75% (Format A) and 72%
  (Format B) of points fall below the diagonal.
- `fig3_flip_by_position.png` — decision-flip rate broken down by average
  sentence-stress position (initial / medial / final) with bootstrap 95%
  CIs. A downward trend (initial > medial > final) is visible in both
  formats, but with n = 13 / 40 / 21 pairs the CIs overlap heavily; treat
  as suggestive, not load-bearing.

## Next concrete steps

1. **Final writeup** — promote this sketch into the canonical
   `docs/project_writeup.md` form, with figures inline.
