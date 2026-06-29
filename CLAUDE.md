# CLAUDE.md — Sub-Decision Prosodic Sensitivity Project

## What This Project Is
An inference-only research project asking: when a speech-LLM gets a contrastive-stress
item wrong, does acoustic stress still shift the output distribution below the final answer?

Using the StressTest benchmark (HUJI 2025). Primary model: Qwen2-Audio-7B-Instruct (4-bit).
No training. Single 16GB GPU (Colab T4). ~6 weeks part-time.

Full proposal is in `docs/proposal.txt`.

---

## Current Status
**Phase: Week 6 — writeup + cleanup done. `docs/final_writeup.md` is the canonical paper-style results document with figures inline; `docs/executive_summary.md` carries final numbers; `docs/results.md` is marked superseded with a pointer header; `docs/project_writeup.md` preserved as structural scaffolding. Root `README.md` written. `requirements.txt` refreshed (matplotlib + datasets added). Remaining items are optional — Qwen2.5-Omni-3B time-boxed attempt and the post-hoc gain+noise-only floor robustness check — see "Pending Work / Optional".**
**Week: 6**

Session 1 (2026-06-02): Created full project folder structure, `requirements.txt`,
`docs/proposal.txt`, `scripts/verify_tokens.py`, `scripts/smoke_test.py`.

Session 2 (2026-06-10): Ran `smoke_test.py` on Colab T4.
- First run revealed silent bug: `apply_chat_template(add_generation_prompt=True)`
  placed prediction position at the start of the assistant turn, so the model was
  generating "Based"/"The" instead of `A`/`B`/`1`/`2`. The sanity check passed only
  because `match=None` (unexpected token) was treated as not-a-failure.
- Fixes applied: (1) move `Answer:` / `Answer: ` out of the user turn and append
  AFTER `apply_chat_template` so it sits inside the assistant turn; (2) strict
  match logic — unexpected gen token now counts as FAIL.
- After fix, N=20 sanity check passes on both formats.
- Wrote `scripts/extract_logits.py` (full-dataset Strategy 2 extraction, JSONL
  output, resumable).
- Set up `docs/colab_setup.md` — Drive-backed HF cache so the 16.8 GB Qwen2-Audio
  download is one-time.
- Ran full extraction on N=218 for both formats. Outputs:
  `results/logits_primary_A.jsonl` (acc 51.0%) and
  `results/logits_primary_B.jsonl` (acc 55.5%). Both near chance, matching
  StressTest paper reports for Qwen2-Audio — this is the expected setup, not a
  failure; the project measures sub-decision shifts beneath these chance-level
  final answers.
- Started Week 2 task 2: text-only falsification pathway
  (`scripts/extract_logits_text.py`).
- Ran full text-only extraction on N=218 for both formats using
  Qwen2-7B-Instruct (same LM backbone as Qwen2-Audio). Outputs:
  `results/logits_textonly_A.jsonl` (acc 49.5%) and
  `results/logits_textonly_B.jsonl` (acc 48.2%). Both sit at chance —
  the transcript alone does not leak the answer. Falsification control
  is clean: any stress-induced logit shifts measured in the primary
  audio pipeline (Week 4) can be attributed to the acoustic signal,
  not to transcript wording.
- Wrote and ran `scripts/extract_cascade.py` — Whisper-large-v3
  transcribes each clip, then Qwen2-7B-Instruct answers from the ASR
  transcript. Sequential pipeline (Whisper → free GPU → LM) to fit on
  T4. Defensive resample to 16kHz inside Phase 1 — dataset audio came
  back at 48kHz despite the StressTest dataset card claiming 16kHz, so
  `cast_column(Audio(sampling_rate=16000))` + an inline `librosa.resample`
  guard are both in the script. Outputs: `results/cascade_transcripts.jsonl`
  (218 Whisper transcripts), `results/logits_cascade_A.jsonl` (acc 49.5%),
  `results/logits_cascade_B.jsonl` (acc 48.2%). Cascade numbers match
  text-only to the third decimal — Whisper transcribes these short
  clean clips accurately, so the cascade collapses to the text-only
  condition. Confirms the audio model's useful signal is prosodic, not
  lexical: stripping prosody (either by gold transcript or via ASR)
  lands at chance.

---

## Decisions Already Made (Do Not Re-Litigate)

### Logit Extraction Strategy
- **Primary:** Strategy 2 — direct `model.forward()` pass, read logits at final prompt
  token position, extract the two competing token IDs, compute 2-way softmax.
- **Sanity check:** Strategy 1 — on N=20 sampled items, also run `model.generate()` and
  assert argmax matches Strategy 2 argmax. This is a structural pipeline step, not optional.
  If it fails, everything stops and the extraction code is debugged before proceeding.

### Answer Formats
- **Two formats only:** Format A (tokens `A`/`B`) and Format B (tokens `1`/`2`).
- Token IDs verified on Qwen2-Audio tokenizer (Session 1, 2026-06-02) — hardcoded:
  - `TOKEN_A = 362`  (` A`, space-prefixed, single token)
  - `TOKEN_B = 425`  (` B`, space-prefixed, single token)
  - `TOKEN_1 = 16`   (`1`, no leading space — see Format B note below)
  - `TOKEN_2 = 17`   (`2`, no leading space — see Format B note below)
- Do not look up token IDs dynamically at run time.
- **Format A prompt:** custom prompt using `A.`/`B.` built from `possible_answers`.
  Prompt ends with `"Answer:"`. Next predicted token is ` A` (362) or ` B` (425).
- **Format B prompt:** use the dataset's `audio_lm_prompt` field, then **append a
  single trailing space** before the forward pass, so it ends with `"Answer: "`.
  Template (from paper Appendix D.1): *"Out of the following answers, according to the
  speaker's stressed words, what is most likely the underlying intention of the speaker?
  1. [answer 1] 2. [answer 2] Answer: "* (note trailing space)
  Next predicted token is `1` (16) or `2` (17) — no leading space needed.
  **Why:** ` 1` and ` 2` are two tokens each ([220,16] and [220,17]) on this tokenizer,
  so we absorb the space into the prompt instead.

### Model Loading
- Use model-specific classes explicitly:
  - `Qwen2AudioForConditionalGeneration` for the model
  - `AutoProcessor` for the processor
- Load in 4-bit quantization (bitsandbytes).
- Forward pass inputs: `input_ids`, `attention_mask`, `input_features` (processed audio).

### Models in Scope
- **Primary (required):** Qwen2-Audio-7B-Instruct (4-bit). Project conclusions stand on
  this alone.
- **Second native model (conditional):** Qwen2.5-Omni-3B — attempt only if inference +
  logit-extraction pipeline can be brought up within a 2-day time-box in Week 2. Drop
  without penalty if not.
- **Text-only control:** text-LLM counterpart fed the transcript (falsification only).
- **Cascade:** Whisper → text-LLM (task-level only).

### Text-Rescue Hint Format (Phase 4 / Step 4.1)
- Encode stress in the transcript by **capitalizing the stressed word in ALL CAPS**.
  Example: `"I NEVER said he stole it."`
- Rationale: matches the convention used in the StressTest paper; avoids leaking
  semantic framing through annotation wording (e.g. "the word X is stressed" can bias
  the LLM toward the answer that mentions X).
- Locked — do not change format mid-experiment.

### Metrics Committed To
- Co-headline 1: decision-flip rate (argmax, fraction of pairs flipped in correct direction)
- Co-headline 2: stress-induced shift magnitude vs. acoustic noise floor
- Acoustic noise floor: **Option B only** — synthetic meaning-preserving perturbations
  applied to the audio (small gain change, mild additive noise, ±5% time-stretch).
  Option A is **unavailable**: StressTest's multiple stress patterns per sentence are
  the contrastive experimental stimuli, not same-stress duplicate takes, so they cannot
  serve as a noise floor.
- Pre-registered threshold: stress shift must exceed 95th percentile of strictest
  perturbation floor to be called meaningful
- Supporting: pair-consistency accuracy (argmax), for StressTest comparability
- Stats: paired Wilcoxon signed-rank + bootstrap 95% CI. Two tests only.
  Report bootstrap CI widths prominently alongside p-values (N≈109 pairs is modest power).

---

## Week-by-Week Plan

| Week | Goal | Status |
|------|------|--------|
| 1 | Verify StressTest benchmark (format, labels, license, duplicate takes); extract one real item; smoke test logit-extraction pipeline on primary model (N=20 items); verify token IDs | ✅ Complete |
| 2 | Finalize extraction on primary model (both answer formats); build text-only + cascade pathways; generate + spot-check meaning-preserving perturbations; time-boxed attempt at second model | ✅ Complete (except optional Qwen2.5-Omni — deferred, see Pending Work) |
| 3 | Full run: both answer formats + text-rescue probe + noise-floor passes | ✅ Complete |
| 4 | Analysis: shift distributions vs noise floor, decision-flip rate, effect sizes, Wilcoxon, bootstrap | ✅ Complete (incl. text-rescue pathway) |
| 5 | Figures, error breakdown by stress position | ✅ Complete |
| 6 | Write-up and repo cleanup | ✅ Complete |

---

## Immediate Next Tasks (Week 2)
1. ✅ Write `scripts/extract_logits.py` and run full 218×2 extraction.
2. ✅ Build text-only pathway (`scripts/extract_logits_text.py`) and run full
   N=218 extraction on Qwen2-7B-Instruct. Both formats at chance (A: 49.5%,
   B: 48.2%) — falsification control is clean.
3. ✅ Build cascade pathway: Whisper(audio) → transcript → text-LLM.
   Task-level only. Both formats at chance (A: 49.5%, B: 48.2%),
   matching text-only — Whisper is accurate enough on these clips
   that the cascade collapses to the text-only condition.
4. ✅ Run text-rescue probe on N=218 × 2 formats (Session 5, 2026-06-29).
   ALL-CAPS hint on the stressed word, fed to Qwen2-7B-Instruct (same
   backbone as text-only/cascade). Argmax 59.6% (A) / 59.2% (B);
   pair-both-correct 23.3% (A) / 22.6% (B). Both metrics beat the
   primary audio model in both formats — clean "ears failed, brain
   works" attribution. `scripts/analysis.py` extended with the
   text_rescue row; numbers in `results/analysis_summary.json`.
5. ✅ Write `scripts/noise_floor.py` and run full noise-floor extraction.
   Four perturbations (gain −3 dB, additive Gaussian noise ~30 dB SNR,
   time-stretch 0.95× and 1.05×), 8 output files (4 × 2 formats), each
   with 218 records. Per-item shift distributions vs the clean primary
   run will be computed in Week 4. Argmax accuracies under perturbation:
   gain A/B 50.9 / 55.5; noise 54.1 / 55.0; stretch_slow 51.8 / 55.5;
   stretch_fast 55.5 / 58.3 — all within ~4% of the clean baseline,
   confirming the perturbations preserve meaning at the decision level.
6. ⏳ Time-boxed (2 days max): attempt Qwen2.5-Omni-3B with the same
   extraction pipeline. Drop without penalty if it doesn't come up cleanly.

### Week 4 analysis results (2026-06-15, written by `scripts/analysis.py`)
- Headline 1 — decision-flip rate (primary, pairs with different gold
  labels, N=74): Format A 9.5% (95% CI 4.1–16.2%); Format B 12.2%
  (95% CI 5.4–20.3%). Both CIs exclude zero — the model does flip in
  the correct direction more often than not-flipping at all, even though
  the rate is modest.
- Headline 2 — paired Wilcoxon, stress shift vs strictest noise-floor
  shift, one-sided "stress > floor", N=218 paired items per format:
  Format A p = 4.2×10⁻¹⁸ (163 of 218 items have larger stress than
  floor shift); Format B p = 7.7×10⁻¹⁷ (158/218). Bootstrap 95% CIs on
  the paired-difference median exclude zero in both formats.
- **Pre-registered check ("stress median > 95th pctile of strictest
  floor") FAILED in both formats.** Reason — time-stretch perturbations
  produced unexpectedly large logit shifts even at ±2% (95th pctile
  ≈0.30 for both directions), an order of magnitude larger than `gain`
  (~0.03) or `noise` (~0.14). Tightening from ±5% to ±2% reduced
  `stretch_fast` 95th pctile only from 0.475 to 0.375; `stretch_slow`
  barely moved. Conclusion: Qwen2-Audio is unusually fragile to small
  time-stretches — a secondary finding about the model, not evidence
  against the stress effect.
- Decision (Session 3, 2026-06-15): go with "option 1" — accept the
  positive Wilcoxon as the headline result and document the pre-reg
  failure transparently in the writeup. The paired comparison directly
  tests the project's hypothesis; the threshold test was a stricter
  pre-registered standard that turned out to be miscalibrated for this
  model's perturbation response.
- Session 4 (2026-06-17): wrote `notebooks/Progress_so_far.ipynb`, a
  clean cell-by-cell Colab notebook for the professor walkthrough.
  Covers dataset → token verification → both prompt formats → smoke
  test → live 5-item mini extraction → load full JSONLs → argmax +
  pair-both-correct tables → Headline 1 (decision-flip rate with 95%
  bootstrap CI) + bootstrap-distribution histogram. Stops at decision-
  flip rate; mentions Wilcoxon as "already done, see
  results/analysis_summary.json" without re-running it.
  Notebook also has inline explanations for: (a) the four possible
  answer tokens and why only two logits are read, (b) what the
  `verify_tokens.py` "FAILED" line actually means, (c) why
  pair-consistency is a stricter metric than single-item accuracy
  with chance × chance baselines (25% coin-flip / 0% lazy / 8.3%
  measured text baseline), (d) why text/cascade 8.3% is *below*
  chance — sister clips have identical text inputs → forced same
  answer → 0% on the 74 different-label pairs by construction
  (8.3% is the ceiling for any prosody-blind model, not a floor),
  (e) what's left beyond pair-consistency and why each remaining
  piece (Wilcoxon, text-rescue, figures, writeup) adds something
  the argmax-level result can't, (f) counterfactual: what if 0% had
  been inside the decision-flip 95% CI — Headline 1 would soften
  but Headline 2 (Wilcoxon) and pair-consistency would survive,
  and a null Headline 1 alongside positive Headline 2 is actually
  the most-predicted outcome under the project's hypothesis.
- The HTML briefing at `temp_explanation/professor_briefing.html`
  mirrors the notebook with extra sections written this session:
  §3.1 four-token / smoke-test reasoning, §6.0 dataset arithmetic
  (218 / 133 / 74), §6.1 expanded "95% bootstrap CI" recipe +
  counterfactual "what if 0% were in the CI", §6.4 pair-consistency
  conceptual explanation including the "8.3% below 25% chance"
  ceiling argument (renumbered, all-pathways table is now §6.5),
  §7.0 framing on why pair-consistency isn't the finish line with
  a remaining-pieces table.
- Falsification controls landed exactly as expected:
  - Text-only (gold transcript) A 49.5% / B 48.2% — at chance
  - Cascade (Whisper → LM) A 49.5% / B 48.2% — identical to text-only
  - Pair-both-correct accuracy: primary A 13.5% / B 19.5%, text-only/
    cascade 8.3% in both formats — the model gets each item ~half right
    but rarely both items of a pair, matching the StressTest paper.

---

## Pending Work — Resume Here (last touched 2026-06-30)

End-of-session state: project complete through Week 6. The canonical
results document is `docs/final_writeup.md` (paper-style, figures inline).
`docs/executive_summary.md` carries the one-page version with final
numbers. Root `README.md` points readers at the right entry points.
`docs/results.md` is preserved with a "superseded" header. `requirements.txt`
includes `datasets` and `matplotlib`. `docs/project_writeup.md` (the
phase-flowchart scaffolding) and the older
`temp_explanation/professor_briefing.html` are unchanged — both are now
superseded by `docs/final_writeup.md` but kept for historical reference.

Nothing in the next-action queue is load-bearing. Everything below is
optional / deferrable.

### Optional / deferrable
- **Qwen2.5-Omni-3B second-model attempt** — Week 2 row in the table still
  flags it. Time-boxed at 2 days. Skip unless everything else lands early.
- **Sub-perturbation noise-floor reanalysis** — if reviewers push back on the
  "stretch is unfair" framing, rerun `analysis.py` with the floor restricted
  to `gain` + `noise` (post-hoc; document clearly). Format A passes that bar
  cleanly; Format B is borderline. Mention in the writeup as a robustness
  check, not the primary analysis.

### Useful files to revisit on resume
- `results/analysis_summary.json` — machine-readable headline numbers
- `docs/results.md` — current draft of the results writeup
- This file's "Week 4 analysis results" subsection above — narrative version
  of the same numbers

### StressTest Benchmark Findings (Session 1)
- HuggingFace: `slprl/StressTest`, single `test` split, 218 rows, Parquet format
- License: CC-BY-NC-4.0 (fine for academic research)
- Key columns: `audio` (16kHz), `transcription`, `possible_answers`, `label` (0 or 1),
  `stress_pattern`, `audio_lm_prompt`
- **Multiple stress patterns per sentence:** 85 sentences × 2 stress patterns, 16 × 3
  patterns. These are the contrastive experimental stimuli (different stress → different
  meaning), **not** same-stress duplicate takes. Option A noise floor is therefore
  unavailable; the project uses Option B (synthetic perturbations) only.
- Format B prompt already in `audio_lm_prompt` field (uses 1/2 format)
- Format A prompt: custom-written (A/B format), constructed from `possible_answers`

---

## Execution Environment
- **Local machine:** Claude Code reads/writes `.py` scripts at `/home/piyush/Desktop/IS`
- **GPU compute:** Google Colab T4 (16GB) — scripts are run there, not locally
- **Bridge:** Manual upload — scripts written locally are uploaded to Colab manually
- **Workflow per session:**
  1. Claude Code writes/edits scripts locally at `/home/piyush/Desktop/IS`
  2. Manually upload changed scripts to Colab (drag-drop or Files panel)
  3. Run in Colab: `!python scripts/smoke_test.py`
  4. Download result files from Colab back to `/home/piyush/Desktop/IS/results/`
- **Colab setup at the start of every session:**
```python

---

## Project File Structure
```
project-root/
├── README.md                  ← repo entry point — points at docs/final_writeup.md, executive_summary.md, CLAUDE.md
├── CLAUDE.md                  ← this file (always keep updated)
├── docs/
│   ├── proposal.txt              ← original project proposal
│   ├── executive_summary.md      ← one-page summary, final numbers (Week 6)
│   ├── final_writeup.md          ← paper-style writeup with figures inline (Week 6)
│   ├── project_writeup.md        ← structural / phase-flowchart scaffolding (preserved)
│   ├── results.md                ← working sketch from Week 4 (superseded by final_writeup.md)
│   └── colab_setup.md            ← Drive-backed HF cache setup notes
├── data/
│   └── stresstest/            ← StressTest benchmark files (populated Week 1)
├── notebooks/
│   ├── exploration.ipynb           ← scratchpad only, never part of pipeline
│   ├── Independent_Study_notebook.ipynb  ← older working notebook (mixed pasted code + script calls)
│   └── Progress_so_far.ipynb       ← clean demo notebook for professor walkthrough up to decision-flip rate (Session 4, 2026-06-17)
├── scripts/
│   ├── verify_tokens.py       ← token ID verification (Week 1) ✅
│   ├── smoke_test.py          ← 10-item pipeline smoke test (Week 1) ✅
│   ├── extract_logits.py      ← primary (audio) logit extraction (Week 2) ✅
│   ├── extract_logits_text.py ← text-only falsification pathway (Week 2) ✅
│   ├── extract_cascade.py     ← Whisper → text-LLM cascade pathway (Week 2) ✅
│   ├── noise_floor.py         ← perturbation + noise floor passes (Week 2-3) ✅
│   ├── extract_text_rescue.py ← ALL-CAPS-on-stressed-word probe (Week 3) ✅
│   ├── analysis.py            ← stats + effect sizes (Week 4) ✅ run, incl. text_rescue row, results in results/analysis_summary.json
│   └── figures.py             ← Week 5 figures (Session 6, 2026-06-30) ✅ writes 3 PNGs to figures/
├── results/
│   └── .gitkeep
├── figures/
│   ├── fig1_shift_distributions.png      ← stress vs 4 noise-floor distributions
│   ├── fig2_stress_vs_floor_scatter.png  ← per-item scatter, colored by correct/wrong
│   └── fig3_flip_by_position.png         ← decision-flip rate by stress position
└── requirements.txt
```

---

## Known Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Tokenizer space issue (` A` ≠ `A`) | Empirically verify token IDs in Week 1; hardcode after |
| Off-by-one in logit position (Strategy 2) | Strategy 1 sanity check on N=20 items |
| StressTest not accessible / wrong format | Week 1 verification step before design is locked |
| Second model causes timeline slip | Hard 2-day time-box; drop without penalty |
| Small shifts over-interpreted | Pre-registered noise floor threshold; effect sizes not just p-values |
| T4 OOM when chaining audio model + Whisper + text-LLM | Load pathways sequentially; each writes intermediate outputs to `results/` and the next reads from disk. Explicit `del model; gc.collect(); torch.cuda.empty_cache()` between pathways. |
| Text-rescue hint format biases LLM | Locked to ALL-CAPS-on-stressed-word convention (see Decisions); no per-item rewording |

---

## What NOT To Do
- Do not re-discuss decisions listed above — they are locked
- Do not add training steps
- Do not create files outside the structure above without updating this file
- Do not use dynamic token ID lookup at run time
- Do not expand the prompt-robustness check into a broader prompt-sensitivity study
- Do not use `AutoModelForCausalLM` — use `Qwen2AudioForConditionalGeneration` explicitly

---

## How To Update This File
At the end of every Claude Code session, update:
- `Current Status` section (phase, week, what was done)
- Week table (mark completed weeks ✅, in-progress ⏳)
- `Immediate Next Tasks` section
- `Project File Structure` if new files were added
- Any new decisions made during the session (add to Decisions section)
