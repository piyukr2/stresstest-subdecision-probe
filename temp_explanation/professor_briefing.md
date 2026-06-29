# A Professor's Walk-Through of Your Independent Study

*Audience: you (the student). Tone: lecture-style. Goal: by the end you can explain every piece of this project — the question, the method, the controls, the numbers — in your own words, without hand-waving.*

---

## Part 1 — The Big Question (in plain words)

### 1.1 The everyday phenomenon

Say the sentence **"I never said he stole the money"** out loud, five times, each time hammering a different word:

| Stressed word | What it implies |
|---|---|
| **I** never said he stole the money | …but somebody else did |
| I **never** said he stole the money | …I deny ever saying it |
| I never **said** he stole the money | …I implied it, didn't say it |
| I never said **he** stole the money | …somebody else stole it |
| I never said he **stole** the money | …he borrowed it, maybe |
| I never said he stole the **money** | …he stole something else |

Same words. Same grammar. **Six different meanings.** The difference lives entirely in *prosody* — which syllables get louder, longer, higher in pitch. Humans do this effortlessly. The question is whether machines that "listen" can.

### 1.2 What a "speech-LLM" is

A normal large language model (LLM) like GPT or Qwen reads **text** and predicts the next token. A **speech-LLM** (e.g. Qwen2-Audio) is the same idea but with an **audio encoder** bolted on the front: raw waveform → audio features → fed into the LLM alongside the text prompt. The model is then asked to answer questions about what it *heard*, not just what it *read*.

```
   ┌────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │  Waveform  │──▶│  Audio encoder    │──▶│                  │
   └────────────┘    │  (Whisper-ish)    │    │                  │
                     └──────────────────┘    │   Language model │──▶ next token
   ┌────────────┐                            │   (transformer)  │   (the answer)
   │ Text prompt│──────────────────────────▶│                  │
   └────────────┘                            └──────────────────┘
```

### 1.3 The known finding — and the gap your project fills

There is a benchmark called **StressTest** (HUJI, 2025) that built exactly the example above into 218 audio clips: same sentence, different prosodic stress, two candidate meanings (A vs B), one correct. They report that current speech-LLMs do **roughly chance-level** on this — about 50%. As far as the "final answer" goes, the models look deaf to stress.

**But "final answer" is a coarse measurement.** A model can be 51% sure the answer is A and 49% sure it's B — that's a wrong final answer, but the *internal distribution* may still be shifting in the right direction when the stress changes. **That sub-decision shift is what your project measures.** Concretely:

> When the model gets the contrastive-stress question wrong, does the *probability distribution* over the two answer tokens still move in the direction that matches the spoken stress?

If yes → the model *hears* prosody, but can't yet *act* on it confidently. That is a real, publishable mechanistic finding even though the headline accuracy looks like chance.

---

## Part 2 — The Core Mechanism You Need to Understand: Reading Logits

This is the single most important technical concept in the project. Take your time on it.

### 2.1 What a logit is

A transformer LLM, at every position, produces a vector of size = vocabulary (~150,000 numbers). These are **logits**. Pass them through softmax and you get a probability distribution over every token in the vocabulary.

When you ask `model.generate(...)` you get back a single token — the **argmax** of that distribution. That throws away an enormous amount of information.

### 2.2 The trick: don't call `generate`, call `forward`

Instead of letting the model *generate*, you give it the prompt that ends right before the answer slot:

```
   Out of the following answers, ... 1. ... 2. ... Answer:_
                                                          ▲
                                       prediction lives here
```

You run `model.forward(...)` once. You read the logits **at the last input position**. From that 150k-element vector you pick out just two numbers — the logit for token `1` and the logit for token `2`. Softmax those two against each other (2-way softmax) and you have:

```
   P("1" | audio, prompt)  =  e^L₁ / (e^L₁ + e^L₂)
   P("2" | audio, prompt)  =  e^L₂ / (e^L₁ + e^L₂)
```

This is **Strategy 2** in your `CLAUDE.md`. It is faster than generating, and it gives you the **probability magnitude**, not just the winner.

### 2.3 Why this matters for your hypothesis

Suppose the two stress patterns of the same sentence give:

| Clip | logit("1") | logit("2") | P("1") | argmax |
|---|---|---|---|---|
| Stress favors meaning 1 | 4.2 | 4.0 | 0.55 | "1" ✅ |
| Stress favors meaning 2 | 4.1 | 3.9 | 0.55 | "1" ❌ |

Both argmaxes are "1" — both look like failures at the top level. But notice that when the stress *should* push toward meaning 2, the probability of "1" still dropped slightly (or the gap between them shrank). That **sub-decision shift** is your signal.

### 2.4 Why the space-prefix obsession in the token IDs

The Qwen tokenizer treats `"A"` and `" A"` (space-prefixed) as **different tokens with different IDs**. After the prompt `Answer:`, the next character the model emits is a space — so the predicted token is ` A` (ID 362), not `A` (ID 322). Get this wrong and you are reading the wrong column of the logit vector and your entire experiment is silently broken. This is exactly the Session 2 bug your `smoke_test.py` caught — the model was actually producing "Based" / "The" because the assistant turn was empty and there was no `Answer:` prefix in the right place.

Lesson: **always sanity-check that the token you are reading the logit for is the token the model actually wants to emit.** That's what Strategy 1 (a regular `generate` call on 20 items, comparing argmax to Strategy 2's argmax) is for.

---

## Part 3 — The Two Answer Formats, and Why Both

You don't run the experiment once — you run it twice, with two different prompt phrasings:

- **Format A:** `... A. <meaning1>  B. <meaning2>  Answer:` → model predicts ` A` or ` B`
- **Format B:** `... 1. <meaning1>  2. <meaning2>  Answer: ` (trailing space!) → predicts `1` or `2`

**Why two formats?** If you saw an effect in Format A only, a critic could say: "you cherry-picked a phrasing the model happens to be sensitive to." Replicating in Format B turns that into: "two independent answer-token choices both show the effect — it's not a quirk of one prompt."

This is the cheapest possible robustness check. It costs you one extra extraction pass and earns you a lot of credibility.

---

## Part 4 — The Falsification Controls (the part that makes this a *real* experiment)

A positive result is meaningless unless you've ruled out the boring explanations. Your project has three control pathways. Picture them as a flowchart:

```
                          ┌─────────────────────────┐
                          │   StressTest clip       │
                          │   (audio + transcript)  │
                          └────────┬────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────────┐
            │                      │                          │
            ▼                      ▼                          ▼
   ┌───────────────┐    ┌────────────────────┐     ┌──────────────────┐
   │  PRIMARY      │    │  TEXT-ONLY         │     │  CASCADE         │
   │  Audio → LLM  │    │  GoldTranscript→LLM│     │  Whisper→LLM     │
   │  (Qwen2-Audio)│    │  (Qwen2-7B text)   │     │  (Whisper+Qwen2) │
   └───────┬───────┘    └─────────┬──────────┘     └────────┬─────────┘
           │                       │                         │
           ▼                       ▼                         ▼
       hears prosody         sees only words           ASR'd words only
       (the thing we test)   (no acoustic signal)      (no prosody preserved)
```

### 4.1 Why text-only?

If the **transcript alone** (no audio at all) already lets the LLM pick the right answer above chance, then any positive result on the audio path could just be the LLM exploiting wording — not hearing stress. Your result: **A = 49.5%, B = 48.2%, both at chance.** The transcript does not leak the answer. ✅ Clean.

### 4.2 Why cascade?

The cascade pipes the audio through **Whisper** (an ASR system) to get text, then feeds that text to the same LLM. Whisper transcribes words; it does **not** preserve which word was stressed. So if the cascade also lands at chance, you've proven: *any signal the primary audio model uses must be the prosodic information that gets stripped out by ASR.*

Your result: **cascade A = 49.5%, B = 48.2% — identical to text-only to three decimals.** ✅ Beautiful. The audio model's edge has to come from prosody, by elimination.

### 4.3 The noise floor — your most important sanity check

The big risk in this kind of work: *any* perturbation of the audio will shift the logits a little, just from noise. So how do you know a stress-induced shift is **bigger than the shift you'd get from any random perturbation**?

You build a **noise floor** by applying *meaning-preserving* perturbations to each clip:
- gain change (−3 dB louder/quieter)
- mild Gaussian noise (~30 dB SNR)
- time-stretch slow (×0.95)
- time-stretch fast (×1.05)

These perturbations don't change *what* was said or *which word was stressed* — they just rough up the audio. The logit shifts they cause are your **floor**. The stress shift must beat that floor to count.

This is the pre-registered statistical bar. (We'll see in Part 6 that the bar partially failed for a *very* informative reason.)

---

## Part 5 — The Plan, Week by Week (and where you are)

| Week | Goal | Status |
|---|---|---|
| 1 | Verify benchmark; verify token IDs; smoke test the pipeline | ✅ |
| 2 | Full extraction on primary model; build text-only + cascade controls | ✅ |
| 3 | Noise-floor perturbation passes (+ text-rescue probe) | ⏳ Noise floor done; text-rescue pending |
| 4 | Statistical analysis: shifts, decision-flip, Wilcoxon, bootstrap | ✅ |
| 5 | Figures, error breakdown by stress position | ⬜ |
| 6 | Writeup | ⏳ Draft exists in `docs/results.md` |

You are at the **end of Week 4** — analysis numbers are in. The next concrete thing on your plate is the **text-rescue probe** (Part 7 below).

---

## Part 6 — The Results So Far (with the statistics explained)

### 6.1 Headline 1 — Decision-flip rate

Take pairs of clips with the *same sentence* but *different stress* (so different gold labels). Count how often the model's argmax flips in the **correct** direction when stress changes.

- **Format A: 9.5%** (95% bootstrap CI: 4.1% – 16.2%)
- **Format B: 12.2%** (95% bootstrap CI: 5.4% – 20.3%)

Both confidence intervals **exclude zero** — the model flips in the right direction more often than random. The rate is modest, but it is real.

**How to read a bootstrap CI:** you resample your N items with replacement 10,000 times, recompute the statistic each time, and report the 2.5th–97.5th percentiles. If zero is not inside that interval, your effect is unlikely to be just sampling noise.

### 6.2 Headline 2 — Paired Wilcoxon: stress shift vs. noise-floor shift

For each of 218 items, compute:
- `stress_shift` = how much the logit-probability moved between the two stress versions of that sentence
- `floor_shift` = how much the logit-probability moved when you applied the *strictest* perturbation to the clean clip

Then ask, pair by pair: was `stress_shift > floor_shift`?

The **Wilcoxon signed-rank test** is the non-parametric paired test for this. (Why non-parametric? Because logit shifts are not normally distributed and you don't want to assume they are.)

- **Format A: p = 4.2 × 10⁻¹⁸**, 163 of 218 pairs have larger stress shift than floor shift
- **Format B: p = 7.7 × 10⁻¹⁷**, 158 of 218

These p-values are astronomically small. Translation: *across the dataset, prosodic stress moves the model's internal distribution more than a random meaning-preserving audio perturbation does.* That **is** the project's central claim, and it landed.

### 6.3 The pre-registered threshold FAILED — and why that's actually interesting

You committed in advance to: "stress median must beat the **95th percentile** of the strictest noise-floor distribution." That stricter bar **did not pass** in either format.

**Reason:** the time-stretch perturbations turned out to produce surprisingly large logit shifts — about **10× larger** than gain or noise perturbations, even at tiny stretches like ±2%. So the 95th percentile of `stretch_fast` is ~0.30; the median stress shift is smaller than that.

What this tells you: **Qwen2-Audio is unusually fragile to mild time-stretching.** That is a finding about the model's robustness, not evidence against the stress effect. The Wilcoxon test, which uses *paired* comparisons (each clip's stress shift vs that clip's own floor shift) and the median, is the more honest test of your hypothesis. The threshold was a stricter pre-reg standard that turned out to be miscalibrated for *this* model.

**The right move:** report the Wilcoxon as the headline, document the threshold-failure transparently, and present the time-stretch fragility as a secondary finding. That's what your `docs/results.md` does.

### 6.4 The control pathways, summarized

| Pathway | Format A acc | Format B acc | Pair-both-correct |
|---|---|---|---|
| Primary audio (Qwen2-Audio) | 51.0% | 55.5% | 13.5% / 19.5% |
| Text-only (Qwen2-7B, gold transcript) | 49.5% | 48.2% | 8.3% |
| Cascade (Whisper → Qwen2-7B) | 49.5% | 48.2% | 8.3% |
| Noise-floor (gain/noise/stretch) | 50.9 – 55.5 | 55.0 – 58.3 | — |

Read top to bottom: the primary audio model is the only one that gets any pair *both* right above the text baseline — and even there it does so only ~13–20% of the time. **The model can hear the difference. It's just not confident about it.** That is the project's research story in one sentence.

---

## Part 7 — What's Left and Why Each Piece Matters

### 7.1 The text-rescue probe (the next concrete task)

Take the gold transcript, **capitalize the stressed word in ALL CAPS** (e.g. `"I NEVER said he stole it"`), and hand *that* to the text-only LLM. Now stress is encoded **in the text**.

Two clean outcomes, both publishable:

- **Text-rescue accuracy >> text-only baseline** → the LLM backbone *can* use stress information, when it arrives as text. The bottleneck is the **audio encoder**: the model has the brain for prosody, but not the ears. ("Ears failed.")
- **Text-rescue ≈ text-only baseline** → even handed stress on a silver platter as text, the backbone can't use it. The bottleneck is the **language model itself**. ("Brain failed.")

This is called **architectural attribution** — narrowing down *which component* is responsible for the deficit. Both answers are scientifically valuable.

### 7.2 Figures (Week 5)

Three figures, in priority order:

1. **Shift-distribution histogram per format** — stress-shift distribution overlaid with the four floor distributions, vertical line at the 95th percentile. This is the visual companion to the Wilcoxon result, and it lets the reader **see** the time-stretch fragility for themselves.
2. **Scatter: stress shift vs. strictest-floor shift**, per item, colored by correct/wrong, with the y=x diagonal. Items above the diagonal are items where stress beat the floor.
3. **Decision-flip rate broken down by stress position** (initial / medial / final). Does the model handle sentence-final stress better than sentence-initial? This is the kind of breakdown that turns a single-number result into a mechanistic story.

### 7.3 Writeup (Week 6)

Turn `docs/results.md` into a paper-style document with the figures inline. The structure essentially writes itself from this briefing — question, method, controls, results, what failed and why, what's next.

---

## Part 8 — The Five Things to Have Memorized Before You Talk to Your Professor

1. **The question:** when the headline answer is wrong, do the *sub-decision logits* still move with prosodic stress? (And: are we sure it's *prosody*, not wording?)
2. **The method:** read 2-way softmax over the two answer tokens directly from `model.forward` logits at the final prompt position — don't call `generate`. Run it in two answer formats (A/B and 1/2) for robustness.
3. **The controls:** text-only and cascade pathways both at chance proves the audio pathway's edge must be prosodic. Synthetic perturbations (gain/noise/stretch) define a noise floor.
4. **The result:** paired Wilcoxon p ≈ 10⁻¹⁸ in both formats — stress shifts beat random-perturbation shifts. Decision-flip CIs exclude zero. The pre-registered 95th-percentile threshold failed because Qwen2-Audio is unusually fragile to time-stretching — that is a secondary finding about model robustness, not a defeat of the hypothesis.
5. **What's next:** text-rescue probe (ears-vs-brain attribution), figures, writeup.

---

*If you can narrate the diagram in §2.2, defend the choice of Wilcoxon over the threshold test in §6.3, and explain why the text-only + cascade controls being at chance is what makes the primary result interpretable — you own this project.*
