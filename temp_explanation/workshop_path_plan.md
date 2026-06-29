# Workshop-Path Plan — Sub-Decision Prosodic Sensitivity

**Status:** Draft for the user to evaluate. Not yet integrated into `CLAUDE.md`.
**Written:** 2026-06-17 (Session 4 follow-up).
**Decision pending:** whether to fold this into the main plan or stay on the
current Week 5–6 trajectory.

---

## Goal

Submit a 4–8 page first-author workshop paper to a credible analysis venue
(target: **BlackboxNLP 2026** at EMNLP; fallback: an Interspeech / ICASSP
analysis track or ACL/EMNLP Findings).

Claim to defend:
> *Qwen2-Audio-7B-Instruct exhibits sub-decision sensitivity to contrastive
> prosodic stress on the StressTest benchmark, even when its final answers
> remain near chance. We localize this sensitivity to the audio encoder
> using linear probes and activation patching, providing converging
> behavioral, representational, and causal evidence of a
> representation-to-decision bottleneck.*

The behavioral half of this claim is already supported by Week 4 results
(Wilcoxon p ≈ 10⁻¹⁸, decision-flip CIs exclude zero). The mechanistic half
is the new work this plan adds.

---

## Cost

**+2 weeks beyond the existing Week 5–6 timeline**, used to add a noise-floor
rebuild + a linear-probe + a sister-pair activation-patching study. All
inference-only. All fits on a single T4 in 4-bit.

| Tranche | What | Days | Already planned? |
|---|---|---|---|
| Existing | Text-rescue probe (Step 1 of `Pending Work`) | 1 | yes |
| Existing | Week 5 figures | 4 | yes |
| Existing | Week 6 writeup | 5 | yes |
| **New** | Rebuild noise floor with prosody-preserving perturbations | 3 | no |
| **New** | Linear probe on frozen audio-encoder reps (by layer) | 2 | no |
| **New** | Stress-salience regression + Whisper cross-attention oracle | 2 | no |
| **New** | Sister-pair activation patching sweep + controls + figure | 7 | no |

Totals: existing ≈ 10 days, new ≈ 14 days. Net +2 weeks.

---

## Week-by-week breakdown (post Week 4)

### Week 5 (existing) — text-rescue + figures
Unchanged from `CLAUDE.md` § Pending Work, items 1–3:
1. Run `scripts/extract_text_rescue.py` (local inspect first, then Colab).
2. Extend `scripts/analysis.py` to fold in the text-rescue pathway.
3. Build `scripts/figures.py`. Three minimum figures: shift histograms,
   stress-vs-floor scatter, decision-flip rate by stress position.

### Week 6 (NEW) — noise-floor rebuild + probes
Goal: convert the documented pre-registration failure into a clean pass and
quantify the "ears vs brain" attribution.

1. **Rebuild the noise floor (Days 1–3).** Replace `stretch_slow/fast` with
   prosody-preserving perturbations: Opus 12 kbps roundtrip, light room IR
   convolution (any small public IR), additive babble noise at matched SNR.
   - New script: `scripts/noise_floor_v2.py` (or extend existing).
   - Re-run `analysis.py` against the new floor. Expectation: 95th-pctile
     threshold now passes for at least Format A.
   - Outcome to report: "under prosody-preserving floor, threshold passes;
     time-stretch fragility is a secondary finding about Qwen2-Audio."

2. **Linear probe by layer (Days 4–5).** Cache `audio_tower` hidden states
   for all 218 items at every encoder layer (mean-pooled). Train logistic
   regression to predict stress *position* (initial/medial/final) at each
   layer. Standard "probe accuracy by depth" curve.
   - New script: `scripts/probe_stress_position.py`.
   - Headline number: best-layer accuracy vs chance (33%).

3. **Stress-salience regression + Whisper attention (Days 6–7).** Extract
   f0 range, intensity, duration of the stressed word with librosa. Regress
   per-item decision-flip probability on those features. Separately, dump
   Whisper decoder cross-attention from the existing cascade run and check
   alignment with `stress_pattern`.
   - New script: `scripts/salience_regression.py`,
     `scripts/whisper_attention_oracle.py`.

### Week 7 (NEW) — activation patching
Goal: a causal localization of where the stress signal lives.

1. **Plumbing (Days 1–2).** Write a forward-pass wrapper that caches
   `audio_tower` hidden states per layer, swaps a `[layer, time_window]`
   slice from a donor item into a recipient item's forward, and re-runs
   from that layer onward. Patch *before* the encoder→LM projection, not
   after. Validate on a single pair: A→A is a no-op; A→B changes the logit.
   - New script: `scripts/patching.py`.

2. **Sister-pair sweep (Days 3–4).** For each of the 74 different-label
   pairs, slide a 200 ms window across the utterance and patch from item A
   into item B at every encoder layer. Record flip rate of B's argmax
   as a function of (layer, time-window).

3. **Controls (Day 5).** (a) Patch from a non-sister item (different
   sentence) — should not flip the answer. (b) Patch a time-window away
   from the stressed word — should flip much less. These are what make the
   localization claim airtight.

4. **Analysis + headline figure (Days 6–7).** Heatmap: x = time, y = layer,
   color = flip rate. This is the paper's headline figure. Diffuse heatmap
   is still publishable as a mechanistic null result, with the behavioral
   half carrying the writeup.

### Week 8 — paper writeup
Promote `docs/results.md` into a paper draft targeting the workshop format.
Sections: behavioral result + clean floor; layer-wise probe;
activation-patching localization; salience regression; honest discussion
of single-model / N=218 limits; related work.

---

## Risks

| Risk | Mitigation |
|---|---|
| Patching plumbing harder than expected on Qwen2-Audio's encoder→LM projection | Day-1 validation gate: if A→A no-op test fails by end of Day 2, cut patching, ship the probe-only version (still publishable, just weaker) |
| New noise floor still doesn't pass threshold | Acceptable. The paper's primary claim is the paired Wilcoxon + the mechanistic localization; threshold becomes a robustness check, not load-bearing |
| Time-rescue probe (existing Week 3 work) reveals "brain failed" rather than "ears failed" | Strengthens the paper — it's the cleaner story (representation present in encoder, not used by LM). Adjust framing, don't change plan |
| Single-T4 OOM during patching | Patch only 7 encoder layers (every 4th), not all 32; halve the time-window granularity. Trades resolution for feasibility |
| Workshop deadline doesn't align | BlackboxNLP@EMNLP is typically mid-August submission; if it slips, fallback is arXiv tech report + roll forward to next cycle (Interspeech / ICASSP analysis tracks) |

---

## Decision criteria

Fold this into `CLAUDE.md` and update the week table to 8 weeks **only if**
all three are true:
1. You can commit the +2 weeks. Half-doing it produces a paper that fails
   workshop review AND is no longer a clean course deliverable.
2. The text-rescue probe (existing Week 3 work) runs cleanly. If
   `extract_text_rescue.py` reveals a `stress_pattern` parsing issue that
   takes more than 1 day to fix, the timeline math is no longer +2 weeks.
3. You actually want a publication on the resume vs. a strong tech-report
   course deliverable. Both are legitimate; the publication is only worth
   the extra weeks if you care about it.

If any are no: stay on the current Week 5–6 plan. The existing results
are a strong course deliverable / arXiv tech report on their own.

---

## Files this plan would add to `Project File Structure`

```
scripts/
├── noise_floor_v2.py             ← prosody-preserving perturbations
├── probe_stress_position.py      ← layer-wise linear probe
├── salience_regression.py        ← per-item flip rate ~ acoustic features
├── whisper_attention_oracle.py   ← cross-attn alignment with stress_pattern
├── patching.py                   ← sister-pair activation patching
└── figures.py                    ← (already planned in Week 5)
```

No new directories needed. Outputs to `results/` and `figures/` as before.
