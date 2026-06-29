# Sub-Decision Prosodic Sensitivity in Qwen2-Audio

An inference-only research project investigating whether
Qwen2-Audio-7B-Instruct's hidden probability distribution responds to
acoustic stress on the StressTest benchmark, even when its final answer is
wrong — and if so, where in the architecture the signal is being lost.

## Headline finding

Across 218 StressTest items in two answer formats, the model's
sub-decision logits respond to stress more than to meaning-preserving
acoustic perturbations (paired Wilcoxon p ≈ 10⁻¹⁸ / 10⁻¹⁷). A text-rescue
probe shows the same language-model backbone uses stress effectively when
it arrives as a capitalized word in the transcript — beating the audio
model itself. The bottleneck is the audio→LM interface, not the language
model. "Ears failed, brain works."

## Where to read

- **[`docs/executive_summary.md`](docs/executive_summary.md)** — one page,
  headline numbers, caveats.
- **[`docs/final_writeup.md`](docs/final_writeup.md)** — paper-style
  writeup with the three figures inline. Start here for the full story.
- **[`docs/project_writeup.md`](docs/project_writeup.md)** — original
  phase-by-phase structural plan; useful as scaffolding context, not as
  the results document.
- **[`docs/proposal.txt`](docs/proposal.txt)** — original project
  proposal.
- **[`CLAUDE.md`](CLAUDE.md)** — running session-by-session narrative,
  locked-in decisions, and the canonical project notebook.

## Where to look

- `scripts/` — one Python script per extraction pathway, plus
  `analysis.py` (statistics) and `figures.py` (plot generation).
  See `docs/final_writeup.md` §6 (Reproducibility) for the full file
  map.
- `results/` — one JSONL per pathway × format, plus
  `analysis_summary.json` with the headline statistics.
- `figures/` — three PNGs referenced from `final_writeup.md`.
- `notebooks/` — `Progress_so_far.ipynb` is the clean Colab walkthrough
  up to the decision-flip-rate step.

## How to reproduce

Set up a Python environment from `requirements.txt`. The heavy extraction
runs (Qwen2-Audio, Qwen2-7B, Whisper-large-v3) target a Google Colab T4
(16 GB VRAM); `docs/colab_setup.md` documents the Drive-backed HF cache
that keeps the 16.8 GB Qwen2-Audio download one-time. Analysis and
figure generation are pure CPU and run locally:

```bash
pip install -r requirements.txt
python scripts/analysis.py       # writes results/analysis_summary.json
python scripts/figures.py        # writes figures/*.png
```

Bootstrap seed is `20260615` and noise-perturbation seeds are
deterministic per-item, so numbers are reproducible to the bit.

## License and scope

StressTest is CC-BY-NC-4.0 (academic use). Conclusions are specific to
Qwen2-Audio-7B-Instruct on contrastive lexical stress in English as
operationalized by StressTest, not "speech-LLMs in general."
