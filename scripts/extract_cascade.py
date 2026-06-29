"""
Cascade pathway: Whisper(audio) -> transcript -> Qwen2-7B-Instruct -> answer.

Task-level only: we care about the final answer (argmax accuracy), not
fine-grained logit-shift analysis. Logits/probabilities are still recorded
so the output JSONL schema matches the audio and text-only pathways and
analysis code stays uniform.

Two phases, run sequentially to respect the 16GB T4 budget:

  Phase 1 (--phase transcribe):
      Load Whisper, transcribe every clip, write cascade_transcripts.jsonl,
      then explicitly free GPU memory.

  Phase 2 (--phase answer):
      Load Qwen2-7B-Instruct, read cascade_transcripts.jsonl, run both
      answer formats, write logits_cascade_{A,B}.jsonl.

Default (--phase both) does phase 1 then phase 2 in one process with a
torch.cuda.empty_cache() in between. Either phase can be re-run with
--resume to pick up partial output.

Usage on Colab:
    !pip install transformers bitsandbytes accelerate datasets librosa -q
    !python scripts/extract_cascade.py --phase both --resume \\
        --out-dir "/content/drive/MyDrive/Colab_Notebooks/Independent_Study/results"
"""

import argparse
import gc
import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import librosa
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

TOKEN_A = 362
TOKEN_B = 425
TOKEN_1 = 16
TOKEN_2 = 17

WHISPER_ID = "openai/whisper-large-v3"
LM_ID = "Qwen/Qwen2-7B-Instruct"
DATASET = "slprl/StressTest"

TRANSCRIPTS_FILE = "cascade_transcripts.jsonl"

# Prompts match scripts/extract_logits_text.py exactly so the only difference
# between text-only and cascade is the source of the transcript text
# (gold vs Whisper output).
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


def build_prompt(fmt, transcription, possible_answers):
    tmpl = FORMAT_A_TEMPLATE if fmt == "A" else FORMAT_B_TEMPLATE
    return tmpl.format(
        transcription=transcription,
        ans_a=possible_answers[0],
        ans_b=possible_answers[1],
    )


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
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------- Phase 1: Whisper transcription ----------

def run_transcribe(ds, start, end, out_path, resume):
    done = already_done_indices(out_path) if resume else set()
    if resume and done:
        print(f"[transcribe] resuming — {len(done)} items already in {out_path}")

    print(f"Loading Whisper from {WHISPER_ID} ...")
    processor = AutoProcessor.from_pretrained(WHISPER_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if resume else "w"

    n_done = 0
    with open(out_path, mode, buffering=1) as f:
        for i in range(start, end):
            if i in done:
                continue
            item = ds[i]
            audio = np.asarray(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
            # Whisper requires 16kHz. Resample defensively if the dataset
            # decoder returns anything else, regardless of cast_column.
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            inputs = processor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).to("cuda", dtype=torch.float16)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    language="en",
                    task="transcribe",
                )
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            rec = {
                "idx": i,
                "gold_transcription": item["transcription"],
                "whisper_transcription": text,
                "stress_pattern": item.get("stress_pattern"),
                "possible_answers": list(item["possible_answers"]),
                "label": int(item["label"]),
            }
            f.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 10 == 0 or i == end - 1:
                print(f"[transcribe] {n_done} new  idx={i}  whisper={text!r}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[transcribe] done. new items: {n_done}")


# ---------- Phase 2: text-LLM answers from Whisper transcripts ----------

def load_lm():
    print(f"Loading tokenizer from {LM_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(LM_ID)
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
                f"tokenizer.encode({decoded_str!r}, add_special_tokens=False)={actual}."
            )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("Loading LM in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        LM_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def extract_one(tokenizer, model, text_prompt, fmt):
    tok_first = TOKEN_A if fmt == "A" else TOKEN_1
    tok_second = TOKEN_B if fmt == "A" else TOKEN_2
    suffix = FORMAT_A_SUFFIX if fmt == "A" else FORMAT_B_SUFFIX

    messages = [{"role": "user", "content": text_prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
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


def load_transcripts(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec["idx"]] = rec
    return rows


def run_answer_format(tokenizer, model, transcripts, fmt, out_path, resume):
    done = already_done_indices(out_path) if resume else set()
    if resume and done:
        print(f"[{fmt}] resuming — {len(done)} items already in {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if resume else "w"

    n_done = 0
    n_correct = 0
    indices = sorted(transcripts.keys())
    with open(out_path, mode, buffering=1) as f:
        for i in indices:
            if i in done:
                continue
            tr = transcripts[i]
            prompt = build_prompt(fmt, tr["whisper_transcription"], tr["possible_answers"])
            lf, ls, pf, ps = extract_one(tokenizer, model, prompt, fmt)
            argmax = 0 if pf > ps else 1
            correct = int(argmax == tr["label"])
            n_correct += correct
            n_done += 1

            rec = {
                "idx": i,
                "format": fmt,
                "pathway": "cascade",
                "label": tr["label"],
                "possible_answers": tr["possible_answers"],
                "stress_pattern": tr.get("stress_pattern"),
                "gold_transcription": tr["gold_transcription"],
                "whisper_transcription": tr["whisper_transcription"],
                "logit_first": lf,
                "logit_second": ls,
                "p_first": pf,
                "p_second": ps,
                "argmax": argmax,
                "correct": correct,
            }
            f.write(json.dumps(rec) + "\n")
            if n_done % 10 == 0 or i == indices[-1]:
                acc = n_correct / n_done
                print(
                    f"[{fmt}] {n_done} new  idx={i}  argmax={argmax}  "
                    f"label={tr['label']}  running_acc={acc:.3f}"
                )

    if n_done:
        print(f"[{fmt}] done. new items: {n_done}  accuracy on new: {n_correct/n_done:.3f}")
    else:
        print(f"[{fmt}] no new items processed.")


# ---------- Driver ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["transcribe", "answer", "both"],
        default="both",
    )
    parser.add_argument("--format", choices=["A", "B", "both"], default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    transcripts_path = os.path.join(args.out_dir, TRANSCRIPTS_FILE)

    if args.phase in ("transcribe", "both"):
        print(f"Loading dataset {DATASET} ...")
        ds = load_dataset(DATASET, split="test")
        # Whisper's feature extractor requires 16kHz; force-resample on read.
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        end = args.end if args.end is not None else len(ds)
        if args.start < 0 or end > len(ds) or args.start >= end:
            print(f"Bad index range: start={args.start} end={end} dataset_len={len(ds)}")
            sys.exit(1)
        print(f"\n=== Phase 1: Whisper transcribe → {transcripts_path} ===")
        run_transcribe(ds, args.start, end, transcripts_path, args.resume)

    if args.phase in ("answer", "both"):
        if not os.path.exists(transcripts_path):
            print(f"No transcripts file at {transcripts_path}. Run --phase transcribe first.")
            sys.exit(1)
        transcripts = load_transcripts(transcripts_path)
        print(f"Loaded {len(transcripts)} transcripts from {transcripts_path}")

        tokenizer, model = load_lm()
        formats = ["A", "B"] if args.format == "both" else [args.format]
        for fmt in formats:
            out_path = os.path.join(args.out_dir, f"logits_cascade_{fmt}.jsonl")
            print(f"\n=== Phase 2: Cascade Format {fmt} → {out_path} ===")
            run_answer_format(tokenizer, model, transcripts, fmt, out_path, args.resume)


if __name__ == "__main__":
    main()
