# What just happened — `extract_logits.py` first run, explained simply

## What we ran

```
python scripts/extract_logits.py --format A --end 20
```

This was a deliberately small test (20 items, Format A only) before launching
the full 218-item × 2-format job. The point was just: does the pipeline work
end-to-end, and roughly how long does it take?

## What the output is telling us

### 1. The model download took ~56 minutes

```
Downloading (incomplete total...): 100% 16.8G/16.8G [56:15<00:00, 4.54MB/s]
```

This is **one-time**. Colab downloaded the 16.8 GB Qwen2-Audio model from
HuggingFace. As long as the Colab runtime stays alive, you don't pay this cost
again. If the runtime disconnects, it has to re-download — that's just the
nature of Colab's ephemeral filesystem.

The download speed (~4.5 MB/s) is HuggingFace's rate limit on unauthenticated
requests. If you set up an `HF_TOKEN` in Colab secrets, future downloads will
be faster — but it's not urgent.

### 2. The actual inference was fast

```
[A] 10/20  idx=9  argmax=0  label=1  running_acc=0.400
[A] 20/20  idx=19  argmax=0  label=1  running_acc=0.500
[A] done. new items: 20  accuracy on new: 0.500
```

Those two progress lines printed back-to-back, meaning **20 audio items took
only a few seconds of compute**. The full 218×2 pass should finish in
roughly 10–30 minutes of compute time, not hours.

### 3. The model is at chance on this task — and that's the whole point

`accuracy on new: 0.500` means the model got 10 out of 20 right. That's a
coin flip. **This is expected**, not a bug.

The StressTest paper itself reports that Qwen2-Audio performs near chance on
this benchmark when forced to pick A vs. B. That is exactly why this project
is interesting: even when the model gets the final answer wrong, does the
acoustic stress still nudge the underlying probability distribution? That is
what we're going to measure in Weeks 3-4, using the logits this script saves.

In other words: **a near-chance final-answer accuracy is the setup, not the
failure mode.** If the model were 95% accurate, the question would be boring.

### 4. The pipeline is healthy

Everything we cared about validating worked:
- Model loaded in 4-bit without OOM.
- StressTest dataset loaded.
- 20 records were written to `results/logits_primary_A.jsonl`.
- The chat-template bug from yesterday is gone (we already proved that on
  N=20 in the smoke test).
- Per-item processing is fast enough that the full run is comfortably within
  one Colab session.

## What this run does NOT tell us

- **Nothing about the actual research question yet.** We haven't compared
  stress-pattern A vs. stress-pattern B for the same sentence, and we haven't
  run the noise floor, and we haven't run the text-only control. All of that
  is Weeks 3–4. This was a plumbing check.
- **Nothing meaningful about model accuracy.** 50% on N=20 is too small a
  sample to draw any conclusion from, and 50% itself is the expected ballpark
  anyway.

## What to do next

Run the full pass on the same Colab session (so the model stays in memory):

```
!python scripts/extract_logits.py --format both --resume
```

`--resume` makes it skip the 20 Format A items already done.

When it finishes, you should have:
- `results/logits_primary_A.jsonl` with 218 lines
- `results/logits_primary_B.jsonl` with 218 lines

Download both files into `/home/piyush/Desktop/IS/results/` on your local
machine. That gives you the raw material for the Week 3 analysis.
