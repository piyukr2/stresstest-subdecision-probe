"""
Full-dataset logit extraction on Qwen2-Audio-7B-Instruct (4-bit).

Runs Strategy 2 (direct forward pass, read logits at last prompt position) over
the StressTest test split for one or both answer formats. Writes one JSON record
per item to JSONL so the run is resumable if Colab disconnects.

The Strategy 1 sanity check from smoke_test.py is intentionally dropped here:
that check was a one-time plumbing verification, already passed on N=20.

Usage on Colab:
    !pip install transformers bitsandbytes accelerate datasets soundfile -q
    !python scripts/extract_logits.py --format both
    !python scripts/extract_logits.py --format A --start 0 --end 50
    !python scripts/extract_logits.py --format B --resume

Outputs:
    results/logits_primary_A.jsonl
    results/logits_primary_B.jsonl
"""

import argparse
import gc
import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

# ---------------------------------------------------------------------------
# Hardcoded token IDs (verified 2026-06-02 on Qwen2-Audio-7B-Instruct).
# Format A: suffix "Answer:"  → next token is ' A' (362) or ' B' (425)
# Format B: suffix "Answer: " → next token is '1'  (16)  or '2'  (17)
# ---------------------------------------------------------------------------
TOKEN_A = 362
TOKEN_B = 425
TOKEN_1 = 16
TOKEN_2 = 17

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATASET = "slprl/StressTest"

FORMAT_A_TEMPLATE = (
    "Out of the following answers, according to the speaker's stressed words, "
    "what is most likely the underlying intention of the speaker? "
    "A. {ans_a} B. {ans_b}"
)
FORMAT_A_SUFFIX = "Answer:"
FORMAT_B_SUFFIX = "Answer: "


def build_format_a_prompt(possible_answers):
    return FORMAT_A_TEMPLATE.format(
        ans_a=possible_answers[0],
        ans_b=possible_answers[1],
    )


def build_format_b_prompt(audio_lm_prompt):
    p = audio_lm_prompt.rstrip()
    if p.endswith("Answer:"):
        p = p[: -len("Answer:")].rstrip()
    return p


def load_model():
    print(f"Loading processor from {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    gc.collect()
    torch.cuda.empty_cache()
    print("Loading model in 4-bit ...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def extract_one(processor, model, audio_array, sr, text_prompt, fmt):
    """Returns (logit_first, logit_second, p_first, p_second)."""
    assert fmt in ("A", "B")
    tok_first = TOKEN_A if fmt == "A" else TOKEN_1
    tok_second = TOKEN_B if fmt == "A" else TOKEN_2
    suffix = FORMAT_A_SUFFIX if fmt == "A" else FORMAT_B_SUFFIX

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "__local__"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    text = text + suffix

    inputs = processor(
        text=text, audio=audio_array, return_tensors="pt", sampling_rate=sr
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
    """Return set of idx already present in the JSONL file (for --resume)."""
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


def run_format(processor, model, ds, fmt, start, end, out_path, resume):
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
            audio_array = np.array(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
            label = int(item["label"])
            answers = item["possible_answers"]

            if fmt == "A":
                prompt = build_format_a_prompt(answers)
            else:
                prompt = build_format_b_prompt(item["audio_lm_prompt"])

            lf, ls, pf, ps = extract_one(processor, model, audio_array, sr, prompt, fmt)
            argmax = 0 if pf > ps else 1
            correct = int(argmax == label)
            n_correct += correct
            n_done += 1

            rec = {
                "idx": i,
                "format": fmt,
                "label": label,
                "possible_answers": list(answers),
                "stress_pattern": item.get("stress_pattern"),
                "transcription": item.get("transcription"),
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
    parser.add_argument("--end", type=int, default=None,
                        help="exclusive; defaults to dataset length")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true",
                        help="skip indices already in the output JSONL")
    args = parser.parse_args()

    processor, model = load_model()

    print(f"Loading dataset {DATASET} ...")
    ds = load_dataset(DATASET, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    end = args.end if args.end is not None else len(ds)
    if args.start < 0 or end > len(ds) or args.start >= end:
        print(f"Bad index range: start={args.start} end={end} dataset_len={len(ds)}")
        sys.exit(1)

    formats = ["A", "B"] if args.format == "both" else [args.format]
    for fmt in formats:
        out_path = os.path.join(args.out_dir, f"logits_primary_{fmt}.jsonl")
        print(f"\n=== Format {fmt} → {out_path} ===")
        run_format(processor, model, ds, fmt, args.start, end, out_path, args.resume)


if __name__ == "__main__":
    main()
