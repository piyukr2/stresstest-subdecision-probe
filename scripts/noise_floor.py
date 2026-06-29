"""
Noise-floor pass: re-run the primary Qwen2-Audio extraction on
meaning-preserving acoustic perturbations of each clip.

Goal: estimate how much the 2-way answer-token probabilities wiggle for
reasons unrelated to stress — small gain changes, mild background noise,
and ±5% time-stretch. Week 4 uses these distributions as the bar a real
stress-induced shift must clear (pre-registered: stress shift > 95th
percentile of the strictest perturbation's shift distribution).

Perturbations (each treated as its own distribution):
    gain         : -3 dB amplitude scaling
    noise        : additive Gaussian noise at ~30 dB SNR
    stretch_slow : 0.95x time-stretch (librosa, pitch-preserving)
    stretch_fast : 1.05x time-stretch (librosa, pitch-preserving)

Two workflows:

    --inspect N
        Don't load the audio model. Write the first N clean+perturbed clips
        as WAVs to results/perturbed_samples/ so you can spot-check by ear
        or spectrogram BEFORE locking the perturbation parameters.

    (default)
        Load Qwen2-Audio, perturb each clip in-memory, run the same Strategy 2
        forward-pass extraction as scripts/extract_logits.py, and write per-
        perturbation JSONL files: results/logits_noise_{perturb}_{fmt}.jsonl.

Determinism: the additive-noise pass seeds numpy with idx, so a given clip
gets the same noise realization every run. Re-running with --resume is safe.

Usage on Colab:
    !pip install transformers bitsandbytes accelerate datasets soundfile librosa -q

    # Spot-check the perturbations first:
    !python scripts/noise_floor.py --inspect 5 \\
        --out-dir "/content/drive/MyDrive/Colab_Notebooks/Independent_Study/results"

    # Then run the full noise-floor extraction:
    !python scripts/noise_floor.py --format both --resume \\
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
from datasets import Audio, load_dataset

# Same token IDs as the primary pipeline (verified 2026-06-02).
TOKEN_A = 362
TOKEN_B = 425
TOKEN_1 = 16
TOKEN_2 = 17

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATASET = "slprl/StressTest"
SR = 16000

# Perturbation parameters. Locked once spot-checked.
GAIN_DB = -3.0
NOISE_SNR_DB = 30.0
STRETCH_SLOW = 0.98
STRETCH_FAST = 1.02

PERTURBATIONS = ("gain", "noise", "stretch_slow", "stretch_fast")

FORMAT_A_TEMPLATE = (
    "Out of the following answers, according to the speaker's stressed words, "
    "what is most likely the underlying intention of the speaker? "
    "A. {ans_a} B. {ans_b}"
)
FORMAT_A_SUFFIX = "Answer:"
FORMAT_B_SUFFIX = "Answer: "


# ---------- perturbation primitives ----------

def perturb_gain(audio, idx):
    factor = 10.0 ** (GAIN_DB / 20.0)
    return (audio * factor).astype(np.float32)


def perturb_noise(audio, idx):
    rng = np.random.default_rng(seed=idx)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (NOISE_SNR_DB / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def perturb_stretch_slow(audio, idx):
    # rate < 1 → slower / longer; pitch preserved
    return librosa.effects.time_stretch(audio, rate=STRETCH_SLOW).astype(np.float32)


def perturb_stretch_fast(audio, idx):
    return librosa.effects.time_stretch(audio, rate=STRETCH_FAST).astype(np.float32)


PERTURB_FNS = {
    "gain": perturb_gain,
    "noise": perturb_noise,
    "stretch_slow": perturb_stretch_slow,
    "stretch_fast": perturb_stretch_fast,
}


# ---------- inspect mode (no model) ----------

def run_inspect(ds, n, out_dir):
    import soundfile as sf

    samples_dir = os.path.join(out_dir, "perturbed_samples")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Writing {n} clean+perturbed sample sets to {samples_dir} ...")

    for i in range(n):
        item = ds[i]
        audio = np.asarray(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        sf.write(os.path.join(samples_dir, f"{i:03d}_clean.wav"), audio, SR)
        for name, fn in PERTURB_FNS.items():
            y = fn(audio, i)
            sf.write(os.path.join(samples_dir, f"{i:03d}_{name}.wav"), y, SR)
        print(f"  idx={i}  transcription={item['transcription']!r}")

    print(
        "Done. Listen to a few. If anything sounds meaning-changing "
        "(e.g. stress pattern altered, syllables dropped), revisit the "
        "perturbation constants at the top of this script."
    )


# ---------- extract mode ----------

def load_model():
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2AudioForConditionalGeneration,
    )

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


def build_format_a_prompt(possible_answers):
    return FORMAT_A_TEMPLATE.format(
        ans_a=possible_answers[0], ans_b=possible_answers[1]
    )


def build_format_b_prompt(audio_lm_prompt):
    p = audio_lm_prompt.rstrip()
    if p.endswith("Answer:"):
        p = p[: -len("Answer:")].rstrip()
    return p


def extract_one(processor, model, audio_array, text_prompt, fmt):
    import torch

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
        text=text, audio=audio_array, return_tensors="pt", sampling_rate=SR
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


def run_perturb_format(processor, model, ds, perturb_name, fmt, start, end, out_path, resume):
    done = already_done_indices(out_path) if resume else set()
    if resume and done:
        print(f"[{perturb_name}/{fmt}] resuming — {len(done)} items in {out_path}")

    perturb_fn = PERTURB_FNS[perturb_name]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if resume else "w"

    n_done = 0
    n_correct = 0
    with open(out_path, mode, buffering=1) as f:
        for i in range(start, end):
            if i in done:
                continue
            item = ds[i]
            audio = np.asarray(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
            if sr != SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

            audio_p = perturb_fn(audio, i)

            label = int(item["label"])
            answers = item["possible_answers"]
            if fmt == "A":
                prompt = build_format_a_prompt(answers)
            else:
                prompt = build_format_b_prompt(item["audio_lm_prompt"])

            lf, ls, pf, ps = extract_one(processor, model, audio_p, prompt, fmt)
            argmax = 0 if pf > ps else 1
            correct = int(argmax == label)
            n_correct += correct
            n_done += 1

            rec = {
                "idx": i,
                "format": fmt,
                "pathway": "noise_floor",
                "perturbation": perturb_name,
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
                    f"[{perturb_name}/{fmt}] {n_done} new  idx={i}  "
                    f"argmax={argmax}  label={label}  running_acc={acc:.3f}"
                )

    if n_done:
        print(
            f"[{perturb_name}/{fmt}] done. new: {n_done}  "
            f"accuracy on new: {n_correct/n_done:.3f}"
        )
    else:
        print(f"[{perturb_name}/{fmt}] no new items processed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        help="If >0, just write this many clean+perturbed WAVs to results/perturbed_samples/ and exit. Does not load the model.",
    )
    parser.add_argument(
        "--perturbations",
        default=",".join(PERTURBATIONS),
        help=f"Comma-separated subset of {PERTURBATIONS}",
    )
    parser.add_argument("--format", choices=["A", "B", "both"], default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset {DATASET} ...")
    ds = load_dataset(DATASET, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    end = args.end if args.end is not None else len(ds)
    if args.start < 0 or end > len(ds) or args.start >= end:
        print(f"Bad index range: start={args.start} end={end} dataset_len={len(ds)}")
        sys.exit(1)

    if args.inspect > 0:
        n = min(args.inspect, end - args.start)
        run_inspect(ds.select(range(args.start, args.start + n)), n, args.out_dir)
        return

    chosen = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    bad = [p for p in chosen if p not in PERTURB_FNS]
    if bad:
        print(f"Unknown perturbations: {bad}. Valid: {list(PERTURB_FNS)}")
        sys.exit(1)

    processor, model = load_model()
    formats = ["A", "B"] if args.format == "both" else [args.format]

    for perturb_name in chosen:
        for fmt in formats:
            out_path = os.path.join(
                args.out_dir, f"logits_noise_{perturb_name}_{fmt}.jsonl"
            )
            print(f"\n=== Noise floor: {perturb_name} / Format {fmt} → {out_path} ===")
            run_perturb_format(
                processor, model, ds, perturb_name, fmt,
                args.start, end, out_path, args.resume,
            )


if __name__ == "__main__":
    main()
