"""
Text-rescue probe: feed the transcript to Qwen2-7B-Instruct with the
stressed word(s) written in ALL CAPS, and run the same forced-choice
extraction as the text-only pathway.

Interpretation
    audio model: partial sensitivity, low argmax accuracy
       +
    text-rescue: clearly above chance  → audio pathway is the bottleneck.
                                         The LM backbone can use stress when
                                         it arrives in lexical form.
       OR
    text-rescue: still at chance       → backbone can't use stress even when
                                         handed to it explicitly. Bottleneck
                                         is at integration, not perception.

Hint format (locked, per CLAUDE.md):
    Capitalize the stressed word(s) in the transcript. Example:
        "I never said he stole it."  ->  "I NEVER said he stole it."
    No surrounding annotation (avoid leaking the word as a label).

Stress-pattern field handling
    On StressTest (slprl/StressTest), `item["stress_pattern"]` is a dict
    with keys {"binary": [0/1, ...], "indices": [int, ...], "words": [str, ...]}.
    The capitalizer also accepts string / list / int forms for robustness.
    It tries word-string replacement first (case-insensitive, punctuation
    stripped) and falls back to index-based capitalization.

Outputs:
    results/logits_text_rescue_A.jsonl
    results/logits_text_rescue_B.jsonl

Usage on Colab:
    !pip install transformers bitsandbytes accelerate datasets -q
    !python scripts/extract_text_rescue.py --format both --resume \\
        --out-dir "/content/drive/MyDrive/Colab_Notebooks/Independent_Study/results"

Local dry-run (no model load) to inspect a few capitalized transcripts:
    python3 scripts/extract_text_rescue.py --inspect 10
"""

import argparse
import gc
import json
import os
import re
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Same token IDs as the text-only pipeline (verified Session 1).
TOKEN_A = 362
TOKEN_B = 425
TOKEN_1 = 16
TOKEN_2 = 17

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
DATASET = "slprl/StressTest"

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


# ---------- transcript capitalization ----------

WORD_RE = re.compile(r"\b[\w'-]+\b")


def _strip_punct(w):
    return re.sub(r"^[^\w']+|[^\w']+$", "", w)


def _normalize_stress_words(stress_pattern):
    """Return a list of stress-word strings (possibly empty)."""
    if stress_pattern is None:
        return []
    if isinstance(stress_pattern, dict):
        words = stress_pattern.get("words") or []
        return [_strip_punct(str(w)) for w in words if str(w).strip()]
    if isinstance(stress_pattern, str):
        s = stress_pattern.strip()
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        word_parts = [p for p in parts if not p.lstrip("-").isdigit()]
        return [_strip_punct(p) for p in word_parts]
    if isinstance(stress_pattern, (list, tuple)):
        return [_strip_punct(str(p)) for p in stress_pattern if isinstance(p, str)]
    return []


def _normalize_stress_indices(stress_pattern):
    """Return a list of integer word indices (possibly empty)."""
    if stress_pattern is None:
        return []
    if isinstance(stress_pattern, dict):
        idxs = stress_pattern.get("indices")
        if idxs is None:
            # derive from binary mask if present
            binary = stress_pattern.get("binary") or []
            return [i for i, b in enumerate(binary) if int(b) == 1]
        return [int(i) for i in idxs]
    if isinstance(stress_pattern, int):
        return [stress_pattern]
    if isinstance(stress_pattern, str):
        parts = re.split(r"[,\s]+", stress_pattern.strip())
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                pass
        return out
    if isinstance(stress_pattern, (list, tuple)):
        out = []
        for p in stress_pattern:
            if isinstance(p, int):
                out.append(p)
            elif isinstance(p, str):
                try:
                    out.append(int(p))
                except ValueError:
                    pass
        return out
    return []


def capitalize_stressed(transcription, stress_pattern):
    """
    Return (new_transcription, n_changed). n_changed == 0 means we couldn't
    locate any stressed word and the caller should skip / warn.

    Strategy: try string match first (case-insensitive whole-word). If that
    fails, fall back to index-based: word index k -> capitalize the k-th
    word (0-indexed) in the transcript. Words are matched by the simple
    \\b[\\w'-]+\\b regex above.
    """
    indices = _normalize_stress_indices(stress_pattern)
    words = _normalize_stress_words(stress_pattern)

    n_changed = 0

    # Pass 1 (preferred): index-based. Capitalize exactly the k-th word in
    # the transcript so repeated word occurrences are not over-capitalized.
    if indices:
        matches = list(WORD_RE.finditer(transcription))
        if matches:
            chars = list(transcription)
            for k in indices:
                if 0 <= k < len(matches):
                    m = matches[k]
                    for i in range(m.start(), m.end()):
                        chars[i] = chars[i].upper()
                    n_changed += 1
            if n_changed > 0:
                return "".join(chars), n_changed

    # Pass 2 (fallback): word-string match. Used only when no indices are
    # available; may over-capitalize repeated words by construction.
    new = transcription
    for w in words:
        if not w:
            continue
        pattern = re.compile(r"\b" + re.escape(w) + r"\b", flags=re.IGNORECASE)
        def repl(m):
            nonlocal n_changed
            n_changed += 1
            return m.group(0).upper()
        new = pattern.sub(repl, new)
    return new, n_changed


# ---------- prompt building ----------

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


# ---------- model load / extract (mirrors extract_logits_text.py) ----------

def load_model():
    import numpy as np  # noqa: F401  (kept consistent with text-only script)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for name, expected, decoded_str in [
        ("TOKEN_A", TOKEN_A, " A"),
        ("TOKEN_B", TOKEN_B, " B"),
        ("TOKEN_1", TOKEN_1, "1"),
        ("TOKEN_2", TOKEN_2, "2"),
    ]:
        actual = tokenizer.encode(decoded_str, add_special_tokens=False)
        if len(actual) != 1 or actual[0] != expected:
            raise RuntimeError(
                f"Token ID mismatch for {name}: expected {expected}, got {actual}."
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
    import numpy as np
    import torch

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


def run_format(tokenizer, model, ds, fmt, start, end, out_path, resume):
    done = already_done_indices(out_path) if resume else set()
    if resume and done:
        print(f"[{fmt}] resuming — {len(done)} items already in {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if resume else "w"

    n_done = 0
    n_correct = 0
    n_skipped = 0
    with open(out_path, mode, buffering=1) as f:
        for i in range(start, end):
            if i in done:
                continue
            item = ds[i]
            label = int(item["label"])
            answers = item["possible_answers"]
            gold_transcription = item["transcription"]
            stress_pattern = item.get("stress_pattern")

            caps_transcription, n_changed = capitalize_stressed(
                gold_transcription, stress_pattern
            )
            if n_changed == 0:
                n_skipped += 1
                if n_skipped <= 5:
                    print(
                        f"  [skip idx={i}] could not locate stressed word in "
                        f"{gold_transcription!r} given stress_pattern={stress_pattern!r}"
                    )
                continue

            if fmt == "A":
                prompt = build_format_a_prompt(caps_transcription, answers)
            else:
                prompt = build_format_b_prompt(caps_transcription, answers)

            lf, ls, pf, ps = extract_one(tokenizer, model, prompt, fmt)
            argmax = 0 if pf > ps else 1
            correct = int(argmax == label)
            n_correct += correct
            n_done += 1

            rec = {
                "idx": i,
                "format": fmt,
                "pathway": "text_rescue",
                "label": label,
                "possible_answers": list(answers),
                "stress_pattern": stress_pattern,
                "transcription": gold_transcription,
                "caps_transcription": caps_transcription,
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
                    f"[{fmt}] {n_done} new  idx={i}  argmax={argmax}  "
                    f"label={label}  running_acc={acc:.3f}"
                )

    if n_done:
        print(
            f"[{fmt}] done. new items: {n_done}  accuracy on new: "
            f"{n_correct/n_done:.3f}  skipped: {n_skipped}"
        )
    else:
        print(f"[{fmt}] no new items processed.  skipped: {n_skipped}")


# ---------- inspect mode ----------

def run_inspect(ds, n):
    print(f"Inspecting first {n} items — no model load.\n")
    n_changed_total = 0
    n_skipped = 0
    for i in range(min(n, len(ds))):
        item = ds[i]
        tr = item["transcription"]
        sp = item.get("stress_pattern")
        new, n_changed = capitalize_stressed(tr, sp)
        print(f"idx={i}")
        print(f"  stress_pattern : {sp!r}")
        print(f"  before         : {tr}")
        print(f"  after          : {new}")
        print(f"  changes        : {n_changed}")
        print()
        if n_changed == 0:
            n_skipped += 1
        else:
            n_changed_total += 1
    print(
        f"Summary on first {min(n, len(ds))} items: "
        f"{n_changed_total} capitalized, {n_skipped} skipped."
    )


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["A", "B", "both"], default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        help="Print N capitalized transcripts and exit (no model load).",
    )
    args = parser.parse_args()

    print(f"Loading dataset {DATASET} ...")
    ds = load_dataset(DATASET, split="test")
    end = args.end if args.end is not None else len(ds)
    if args.start < 0 or end > len(ds) or args.start >= end:
        print(f"Bad index range: start={args.start} end={end} dataset_len={len(ds)}")
        sys.exit(1)

    if args.inspect > 0:
        n = min(args.inspect, end - args.start)
        run_inspect(ds.select(range(args.start, args.start + n)), n)
        return

    tokenizer, model = load_model()
    formats = ["A", "B"] if args.format == "both" else [args.format]
    for fmt in formats:
        out_path = os.path.join(args.out_dir, f"logits_text_rescue_{fmt}.jsonl")
        print(f"\n=== Text-rescue Format {fmt} → {out_path} ===")
        run_format(tokenizer, model, ds, fmt, args.start, end, out_path, args.resume)


if __name__ == "__main__":
    main()
