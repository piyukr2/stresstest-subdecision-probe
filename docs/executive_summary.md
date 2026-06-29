# Sub-Decision Prosodic Sensitivity in Speech-LLMs
### Executive Summary — One Page

**Question.** When a speech-LLM picks the wrong answer on a contrastive-stress
question, does the speaker's prosody still shift the model's internal answer
distribution — even if it doesn't change the top choice? And if it does, where
in the architecture is the signal being lost?

**Why it matters.** A wrong final answer is a coarse signal. It conflates two
very different failure modes: a model whose audio encoder discards prosody
("the ears failed") and a model whose language backbone receives the prosodic
signal but underweights it ("the brain failed"). These need different fixes.
Reading the model's hidden probabilities — not just its top-1 answer —
separates them.

**Approach.** Inference-only study. No training.
- **Model:** Qwen2-Audio-7B-Instruct, 4-bit quantization, single 16 GB GPU.
- **Benchmark:** StressTest (HUJI 2025), 218 audio items, 74 contrastive
  label-differing pairs.
- **Method:** Forced-choice prompting, halt generation after the first token,
  extract raw logits for the two competing answer tokens, compute a two-way
  softmax. This exposes sub-decision probability shifts that argmax decoding
  hides.

---

## Headline findings

**1. Sub-decision logits respond to stress, even when the final answer is
wrong.** Paired one-sided Wilcoxon (stress shift > strictest noise-floor
shift) on N = 218 per-item observations:
- Format A (A/B options): p = 4.2 × 10⁻¹⁸, 75% of items show stress > floor.
- Format B (1/2 options): p = 7.7 × 10⁻¹⁷, 72% of items show stress > floor.

**2. The model flips its top answer in the correct direction about one pair
in ten.** Decision-flip rate on the 74 label-differing pairs:
- Format A: 9.5%, bootstrap 95% CI (4.1%, 16.2%).
- Format B: 12.2%, bootstrap 95% CI (5.4%, 20.3%).
CIs exclude zero in both formats.

**3. The bottleneck is in the audio→LM interface, not the language model.**
A text-rescue probe — same LM backbone (Qwen2-7B-Instruct), transcript with
the stressed word in ALL CAPS — beats the primary audio model on both
metrics:

| Pathway | A argmax | B argmax | A pair-both | B pair-both |
|---|---|---|---|---|
| Primary audio (Qwen2-Audio) | 50.9% | 55.5% | 13.5% | 19.5% |
| Text-only (Qwen2-7B, no hint) | 49.5% | 48.2% | 8.3% | 8.3% |
| Cascade (Whisper → Qwen2-7B) | 49.5% | 48.2% | 8.3% | 8.3% |
| **Text-rescue (ALL-CAPS hint)** | **59.6%** | **59.2%** | **23.3%** | **22.6%** |

The same backbone that sat at chance on plain transcripts jumps ~10 points
when stress is delivered as a single capitalized word — outperforming the
audio model. "Ears failed, brain works."

---

## Caveats

- **Pre-registered threshold failed.** The stricter pre-reg check (stress
  median > 95th percentile of the strictest floor) did not pass in either
  format, because Qwen2-Audio is unusually fragile to ±2% time-stretch
  perturbations — the strictest floor was wider than the threshold assumed.
  This is itself a secondary finding about the model. The Wilcoxon and
  decision-flip results are paired per-item comparisons and stand on their
  own.
- **N is modest.** 218 items, 74 eligible flip-rate pairs. Bootstrap CIs are
  reported as prominently as p-values.
- **One model, English only.** Conclusions are specific to Qwen2-Audio-7B
  on contrastive lexical stress as operationalized by StressTest. A
  second-model attempt (Qwen2.5-Omni-3B) was time-boxed and dropped.

---

## Methodological commitments (all honored)

- Strategy-1 vs Strategy-2 pipeline sanity check (`generate()` argmax vs
  forward-pass logit argmax) before any analysis.
- Token IDs empirically verified once, then hardcoded — no runtime lookup.
- Text-rescue hint format locked in advance (ALL CAPS on the stressed word,
  no annotations), avoiding post-hoc tuning.
- Bootstrap CIs reported alongside p-values.
- Pre-registered threshold reported transparently as failed, not retuned.

**Deliverable.** A short technical report (`docs/final_writeup.md`), three
figures in `figures/`, raw JSONL outputs in `results/`, and a public,
reproducible Colab-ready codebase. The repository's `CLAUDE.md` carries the
project's running narrative and all locked-in decisions.
