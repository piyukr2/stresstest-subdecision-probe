# Sub-Decision Prosodic Sensitivity in Speech-LLMs
### A Project Write-Up for Flowchart, Mind-Map, and Infographic Generation

---

## The Research Question in One Sentence
When a speech-capable large language model gets a contrastive-stress question
**wrong**, does the acoustic stress in the speaker's voice still shift the model's
internal answer distribution — even if it doesn't shift the final answer?

## Why This Question Matters
Speech-LLMs are increasingly deployed in voice assistants, transcription tools,
and accessibility software. Prior work (StressTest, HUJI 2025) shows these models
often pick the wrong answer when meaning hinges on prosody. But a **wrong final
answer** is a coarse signal. The model might be:

- **Truly deaf to prosody** — its internal probabilities don't move at all.
- **Partially sensitive** — its probabilities shift in the correct direction, but
  not enough to change the top choice.

These two failure modes call for very different engineering fixes. The first
implicates the audio encoder ("the ears"); the second implicates the language
model backbone ("the brain"). This project distinguishes them by reading the
model's hidden probability distribution instead of just its top-1 answer.

## Scope and Constraints
- **Inference only** — no training, no fine-tuning.
- **Primary model:** Qwen2-Audio-7B-Instruct, loaded in 4-bit quantization.
- **Hardware:** single 16 GB GPU (Google Colab T4).
- **Timeline:** ~6 weeks, part-time.
- **Benchmark:** StressTest (slprl/StressTest on HuggingFace) — 218 audio items,
  each a sentence whose meaning depends on which word the speaker stresses.

---

# Phase 1 — Foundation: Setting the Baselines

**Goal:** Build a controlled environment so any signal we attribute to stress is
defensibly *not* an artifact of noise, text, or pipeline bugs.

### Step 1.1 — Verify the Benchmark
Confirm the StressTest dataset loads, has the expected schema (audio at 16 kHz,
two possible answers per item, a ground-truth label), and a usable license
(CC-BY-NC-4.0 for academic work).

**Rationale:** Every downstream measurement assumes the dataset is what we think
it is. A two-hour verification step at the start prevents weeks of analysis
built on misread columns.

### Step 1.2 — Build the Acoustic Noise Floor
Apply small, **meaning-preserving** perturbations to each audio clip — a mild
gain change, low-level additive noise, a ±5% time-stretch — and measure how
much the model's internal probability for the correct answer shifts purely from
those perturbations.

**Rationale:** Any model's output distribution wobbles slightly when the input
changes, even when the change is semantically meaningless. We need to know how
big that "baseline wobble" is before we can claim a stress-induced shift is
real. The pre-registered rule: a stress shift must exceed the **95th percentile**
of the noise-floor distribution to count as meaningful.

> **Design note:** An earlier plan considered using same-speaker duplicate takes
> from the benchmark as the noise floor. On inspection, StressTest's multiple
> versions of each sentence carry *different* stress patterns — they are the
> experimental stimuli, not duplicate takes. So we use synthetic perturbations
> instead, which are also fully reproducible.

### Step 1.3 — Text-Only & Whisper Falsification Checks
Feed the written **transcript** of each audio clip into:
1. A text-only LLM (no audio modality at all).
2. A cascade pipeline: Whisper transcribes the audio → text-LLM answers from
   the transcript.

**Rationale:** If either of these systems can solve StressTest from text alone,
then audio prosody is *not* the only path to the answer, and the experiment is
ill-posed. Both pipelines should perform near chance. This is a falsification
check on the entire study premise.

---

# Phase 2 — Core Experiment: Logit Extraction

**Goal:** Measure not just what the model *says*, but the probability
distribution behind what it says.

### Step 2.1 — Input the Audio Pairs
Each StressTest item is part of a pair: same words, different stressed word,
opposite correct answer. Feed both audio versions through the model.

### Step 2.2 — Strict Forced-Choice Prompting
Ask the model a single forced-choice question with two answer options, and
prompt it so the next predicted token is constrained to one of two specific
tokens. We use two answer formats:

- **Format A:** options labeled `A.` and `B.`; next token is ` A` or ` B`.
- **Format B:** options labeled `1.` and `2.`; next token is `1` or `2`.

The two formats let us test whether any observed effect is genuine prosodic
sensitivity or an artifact of how the options were labeled.

### Step 2.3 — Stop & Extract the Logits
Instead of letting the model generate a full sentence, we halt after the **very
first token** and read the raw logit values for exactly the two competing
answer tokens. A two-way softmax over those two logits gives a clean probability
for each candidate answer.

**Rationale:** A free-form generated answer collapses a rich probability
distribution into a single argmax decision. Reading the logits directly
preserves the sub-decision information that is the whole point of this study.
Restricting the softmax to the two answer tokens (rather than the full
vocabulary) gives a well-defined "probability of A given the question is A or B."

> **Pipeline integrity check:** Independently run `model.generate()` for one
> token on a sample of items and assert that its output token matches the
> argmax of our extracted logits. If they disagree, the extraction code has a
> bug and everything stops until it's fixed.

---

# Phase 3 — Safety Checks & Branching Results

**Goal:** Decide, per item, whether the model exhibited a real reaction to stress.

### Step 3.1 — Calculate the Shift
For each pair, compute how much the probability of the *correct* answer changed
between the two audio versions. This is the "stress-induced shift."

### Step 3.2 — Filter Against the Noise Floor (Branching Point)
Compare the stress-induced shift to the noise-floor distribution from Step 1.2.

- **Path A — Significant Shift:** The shift exceeds the 95th-percentile noise
  threshold. The model *noticed* the stress at the distributional level. If it
  still picked the wrong final answer, this is a **measurable-but-decision-
  insufficient** influence: the signal was there but underweighted.
- **Path B — Negligible Shift:** The shift falls within the noise floor. We
  categorize the model as showing **no measurable influence** for this item.

### Step 3.3 — Format Swap for Robustness
Re-run the entire experiment with Format B (1/2) and check that whatever
directional effect we saw in Format A (A/B) survives. A real prosodic effect
should not depend on whether the answer was called "A" or "1."

---

# Phase 4 — Diagnostic Probe: Pinpointing the Failure

**Goal:** When the model fails (especially Path B items), determine *where* in
the architecture the failure lives.

### Step 4.1 — Text-Rescue Test
Re-present each failed item to the model, but this time encode the stress
**in the transcript** by capitalizing the stressed word in ALL CAPS
(e.g., `"I NEVER said he stole it."`).

- **Outcome A — The "Ears" Failed:** The model now answers correctly. The
  language backbone can use prosodic information when it arrives in text form;
  the bottleneck is the upstream audio pathway (encoder, projector, fusion).
- **Outcome B — The "Brain" Failed:** The model still gets it wrong even with
  the explicit text hint. The limitation lives downstream of the audio
  pathway, in how the language model integrates the stress signal into its
  reasoning.

**Rationale:** This is the diagnostic payoff of the whole study. It converts an
opaque "the model got it wrong" into an actionable architectural attribution.

---

# Phase 5 — Final Verdict: Statistical Proof & Conclusion

**Goal:** Convert per-item measurements into a defensible aggregate claim.

### Step 5.1 — Statistical Testing
- **Wilcoxon signed-rank test** on paired shifts (matched pairs of audio
  stimuli). Chosen because it's non-parametric and appropriate for paired
  data without distributional assumptions.
- **Bootstrap 95% confidence intervals** on the shift magnitude and the
  decision-flip rate, to characterize effect size and uncertainty.

> **Honesty note:** With ~109 pairs, statistical power is modest. Confidence
> interval *widths* will be reported as prominently as p-values, so the
> conclusion is calibrated to the data we actually have.

### Step 5.2 — Final Graded Conclusion
For the primary model (Qwen2-Audio-7B), pair the quantitative finding with the
architectural diagnosis:

| Finding | Combined With | Final Claim |
|---|---|---|
| No measurable influence | Brain failed even with text | Model is largely insensitive to prosody at both audio and reasoning levels |
| No measurable influence | Ears failed, brain succeeded with text | Audio pathway loses prosody; backbone is capable |
| Measurable-but-decision-insufficient | Brain failed with text too | Prosody reaches the model but isn't weighted enough; both layers contribute |
| Measurable-but-decision-insufficient | Brain succeeded with text | Audio pathway transmits a weak signal; backbone uses it when it's stronger |

---

## Headline Outputs of the Study
1. **Decision-flip rate** — fraction of pairs where the model's top answer flips
   in the correct direction when stress flips.
2. **Stress-induced shift magnitude vs. acoustic noise floor** — the distribution
   of sub-decision shifts, plotted against the noise threshold.
3. **Architectural attribution** — per-item categorization of "ears failed"
   vs. "brain failed" from the text-rescue probe.

## What This Study Will *Not* Claim
- It will not claim Qwen2-Audio is universally prosody-blind or prosody-aware.
  The claim is specific to **contrastive lexical stress** as operationalized
  by StressTest.
- It will not claim training-time fixes. The project is inference-only and
  diagnostic.
- It will not generalize beyond English or beyond the speakers in StressTest.

## Risks Acknowledged Up Front
| Risk | Mitigation |
|---|---|
| Tokenizer quirks (e.g. ` A` vs `A` as different tokens) | Token IDs empirically verified once, then hardcoded |
| Off-by-one errors in logit position | Independent generate-vs-forward sanity check on every smoke-test run |
| Modest statistical power (~109 pairs) | Bootstrap CIs reported as prominently as p-values |
| Single-GPU memory pressure when chaining pipelines | Sequential loading; each pathway writes intermediates to disk |
| Text-rescue hint biasing the LLM | Hint format locked in advance (ALL CAPS, no annotations) |

## Timeline (6 Weeks, Part-Time)
| Week | Focus |
|---|---|
| 1 | Verify benchmark; smoke-test the logit-extraction pipeline on 10–20 items |
| 2 | Finalize both answer formats; build text-only + Whisper cascade pathways; generate noise-floor perturbations |
| 3 | Full experiment run on all 218 items: both formats, text-rescue, noise-floor passes |
| 4 | Analysis: shift distributions, decision-flip rate, Wilcoxon, bootstrap |
| 5 | Figures and per-stress-position error breakdown |
| 6 | Write-up and repository cleanup |
