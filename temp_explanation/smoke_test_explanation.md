# Smoke Test: Purpose, Results, Why, and Fix

## Purpose of `scripts/smoke_test.py`

It's a 10-item end-to-end dry run of the core pipeline before we commit to a full experiment. It checks three things:

1. **Can we load the model and dataset on Colab T4?** (memory, quantization, audio decoding all working)
2. **Does our logit-extraction method (Strategy 2) actually work?** We feed audio + prompt into the model and read the probability of the two answer tokens (`A`/`B` or `1`/`2`).
3. **Sanity check:** does Strategy 2 (read logits directly) agree with Strategy 1 (let the model generate one token normally)? If they disagree, our extraction is reading the wrong place and the whole project is built on garbage.

Think of it as: "before we run 218 items × multiple conditions, does the plumbing even work on 10?"

## Current results

- The script printed **"SANITY CHECK PASSED"** — but that's a lie.
- The `gen_A` and `gen_B` columns show the model actually generated words like **"Based"** and **"The"**, not `A`/`B`/`1`/`2`.
- Format B argmax is `2` on 9 of 10 items with extreme confidence (often 99%+). The model looks like it's blindly picking the second option.

## Why this happened

We used `apply_chat_template(..., add_generation_prompt=True)`. That wraps our prompt in Qwen's chat format and ends it like this:

```
<|im_start|>user
...A. xxx B. yyy Answer:<|im_end|>
<|im_start|>assistant
```

So the position the model is predicting from is **the very start of the assistant's reply** — not the token right after `Answer:`. From that position, the model wants to start a normal sentence ("Based on the audio…", "The speaker…"), which is exactly what we see.

Two consequences:

- **Strategy 1** generates "Based" / "The" because that's the natural start of an answer.
- **Strategy 2** reads logits at that same wrong position and forces a 2-way softmax between ` A` and ` B` (or `1` and `2`). Both have near-zero probability there, but one is slightly larger, and the softmax inflates that tiny gap into a 99% number. That's why the model looks ultra-confident and biased toward B/2 — it's noise being amplified.

They "agree" only in the trivial sense that they're both looking at the wrong place.

The script said PASSED because the match logic at `smoke_test.py:225-226` only flags a mismatch when the generated token is one of the four expected IDs. When the model generates "Based" (some other token), `match` is set to `None`, and the summary treats `None` as "not a failure".

## How to fix

**Idea:** put `Answer:` inside the assistant's turn instead of the user's turn. That way, when the model predicts the next token, it's predicting what comes right after `Answer:` — which should naturally be ` A` / ` B` / `1` / `2`.

Two concrete changes:

1. **Prompt construction:** remove `Answer:` from the user message. After `apply_chat_template(..., add_generation_prompt=True)`, manually append `"Answer:"` (Format A) or `"Answer: "` (Format B) to the resulting text. Now the prompt ends inside the assistant turn, right where we want the model to commit to an answer.

2. **Sanity check logic:** treat `match is None` (generated token isn't one of the four expected IDs) as a **FAIL**, not a silent pass. That way a future regression of this same bug can't hide again.

After the fix, on 10 items we should see: `gen_A` column showing "A" or "B", `gen_B` showing "1" or "2", probabilities that aren't all 99%, and the B/2 bias going away (or at least dropping a lot).
