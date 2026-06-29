"""
Text-only falsification pathway.

Feeds the StressTest transcript (no audio, no stress annotation) to the LM
backbone of Qwen2-Audio — Qwen2-7B-Instruct — and extracts the same 2-way
probabilities over the answer tokens that the primary pipeline produces.

Why Qwen2-7B-Instruct specifically: Qwen2-Audio-7B-Instruct is built on
Qwen2-7B-Instruct + an audio encoder. Using the same LM backbone here isolates
the audio modality as the only difference between this control and the primary
run. Same tokenizer family → same hardcoded token IDs (asserted at startup).

Falsification logic: if the primary model's stress-induced logit shifts
(measured in Week 4) disappear in this text-only run, those shifts depend on
acoustic input. If they survive, they're being driven by something in the
transcript (semantic leakage / answer-option wording) — which would weaken the
project's conclusion.

Usage on Colab:
    !pip install transformers bitsandbytes accelerate datasets -q
    !python scripts/extract_logits_text.py --format both --resume

Outputs:
    results/logits_textonly_A.jsonl
    results/logits_textonly_B.jsonl
"""

import argparse
import gc
import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Same token IDs as the primary pipeline (Qwen2 tokenizer family).
# Asserted at startup against the loaded tokenizer.
TOKEN_A = 362   # ' A'
TOKEN_B = 425   # ' B'
TOKEN_1 = 16    # '1'
TOKEN_2 = 17    # '2'

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
DATASET = "slprl/StressTest"

# Prompts mirror the primary pipeline's wording so the only difference is the
# input modality (transcript text vs. audio). The "according to the speaker's
# stressed words" phrasing is kept verbatim — without audio the model has no
# way to know stress, which is exactly what this control measures.
FORMAT_A_TEMPLATE = (
    "Transcript: \"{transcription}\"\n\n"
    "Out of the following answers, according to the speaker's stressed words, "
    "what is most likely the underlying intention of the speaker? "
    "A. {ans_a} B. {ans_b}"
)
FORMAT_B_TEMPLATE = (
    "Transcript: \"{transcription}\"\n\n"
    "Out of the following answers, according to the speaker's stressed words, "
    "what is most likely the underlying intention of the speaker? "
    "1. {ans_a} 2. {ans_b}"
)
FORMAT_A_SUFFIX = "Answer:"
FORMAT_B_SUFFIX = "Answer: "


def build_format_a_prompt(transcription, possible_answers):
    return FORMAT_A_TEMPLATE.format(
        transcription=transcription,
        ans_a=possible_answers[0],
        ans_b=possible_answers[1],
    )


def build_format_b_prompt(transcription, possible_answers):
    return FORMAT_B_TEMPLATE.format(
        transcription=transcription,
        ans_a=possible_answers[0],
        ans_b=possible_answers[1],
    )


def load_model():
    print(f"Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Sanity-check that the Qwen2 tokenizer's IDs match the hardcoded values
    # we verified on Qwen2-Audio. Fail loud if the assumption ever breaks.
    for name, expected, decoded_str in [
        ("TOKEN_A", TOKEN_A, " A"),
        ("TOKEN_B", TOKEN_B, " B"),
        ("TOKEN_1", TOKEN_1, "1"),
        ("TOKEN_2", TOKEN_2, "2"),
    ]:
        actual = tokenizer.encode(decoded_str, add_special_tokens=False)
        if len(actual) != 1 or actual[0] != expected:
            raise RuntimeError(
                f"Token ID mismatch for {name}: hardcoded={expected}, "
                f"tokenizer.encode({decoded_str!r}, add_special_tokens=False)={actual}. "
                f"Qwen2-7B-Instruct tokenizer differs from the verified IDs. "
                f"Stop and re-verify token IDs before continuing."
            )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    gc.collect()
    torch.cuda.empty_cache()
    print("Loading model in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def extract_one(tokenizer, model, text_prompt, fmt):
    """Returns (logit_first, logit_second, p_first, p_second)."""
    assert fmt in ("A", "B")
    tok_first = TOKEN_A if fmt == "A" else TOKEN_1
    tok_second = TOKEN_B if fmt == "A" else TOKEN_2
    suffix = FORMAT_A_SUFFIX if fmt == "A" else FORMAT_B_SUFFIX

    messages = [{"role": "user", "content": text_prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Same trick as primary pipeline: put "Answer:" inside the assistant turn
    # so the next predicted token is the answer token itself.
    text = text + suffix

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_last = outputs.logits[0, -1, :]
    lf = logits_last[tok_first].float().item()
    ls = logits_last[tok_second].float().item()

    m = max(lf, ls)
    ef = np.exp(lf - m)
    es = np.exp(ls - m)
    total = ef + es
    return lf, ls, ef / total, es / total


def already_done_indices(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_format(tokenizer, model, ds, fmt, start, end, out_path, resume):
    done = already_done_indices(out_path) if resume else set()
    if resume and done:
        print(f"[{fmt}] resuming — {len(done)} items already in {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if resume else "w"

    n_total = end - start
    n_correct = 0
    n_done = 0

    with open(out_path, mode, buffering=1) as f:
        for i in range(start, end):
            if i in done:
                continue

            item = ds[i]
            label = int(item["label"])
            answers = item["possible_answers"]
            transcription = item["transcription"]

            if fmt == "A":
                prompt = build_format_a_prompt(transcription, answers)
            else:
                prompt = build_format_b_prompt(transcription, answers)

            lf, ls, pf, ps = extract_one(tokenizer, model, prompt, fmt)
            argmax = 0 if pf > ps else 1
            correct = int(argmax == label)
            n_correct += correct
            n_done += 1

            rec = {
                "idx": i,
                "format": fmt,
                "pathway": "textonly",
                "label": label,
                "possible_answers": list(answers),
                "stress_pattern": item.get("stress_pattern"),
                "transcription": transcription,
                "logit_first": lf,
                "logit_second": ls,
                "p_first": pf,
                "p_second": ps,
                "argmax": argmax,
                "correct": correct,
            }
            f.write(json.dumps(rec) + "\n")

            if n_done % 10 == 0 or i == end - 1:
                acc = n_correct / n_done
                print(
                    f"[{fmt}] {n_done}/{n_total - len(done)}  idx={i}  "
                    f"argmax={argmax}  label={label}  running_acc={acc:.3f}"
                )

    if n_done:
        print(f"[{fmt}] done. new items: {n_done}  accuracy on new: {n_correct/n_done:.3f}")
    else:
        print(f"[{fmt}] no new items processed (all in resume set).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["A", "B", "both"], default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model()

    print(f"Loading dataset {DATASET} ...")
    ds = load_dataset(DATASET, split="test")

    end = args.end if args.end is not None else len(ds)
    if args.start < 0 or end > len(ds) or args.start >= end:
        print(f"Bad index range: start={args.start} end={end} dataset_len={len(ds)}")
        sys.exit(1)

    formats = ["A", "B"] if args.format == "both" else [args.format]
    for fmt in formats:
        out_path = os.path.join(args.out_dir, f"logits_textonly_{fmt}.jsonl")
        print(f"\n=== Text-only Format {fmt} → {out_path} ===")
        run_format(tokenizer, model, ds, fmt, args.start, end, out_path, args.resume)


if __name__ == "__main__":
    main()
