"""
Verify token IDs for answer tokens on Qwen2-Audio-7B-Instruct tokenizer.

Run on Colab after loading the model:
    !python scripts/verify_tokens.py

Prints token IDs for ' A', ' B', ' 1', ' 2' (with leading space) and their
no-space variants. Single-token status is critical — if any answer token
is NOT a single token, logit extraction at position [-1] will be wrong.
"""

from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

print(f"Loading tokenizer from {MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

CANDIDATES = {
    "A":  "A",
    "B":  "B",
    "1":  "1",
    "2":  "2",
    " A": " A",
    " B": " B",
    " 1": " 1",
    " 2": " 2",
}

print("\n" + "=" * 55)
print(f"{'String':<8} {'Token IDs':<25} {'Single token?'}")
print("=" * 55)

problems = []
for label, text in CANDIDATES.items():
    ids = tokenizer.encode(text, add_special_tokens=False)
    is_single = len(ids) == 1
    flag = "" if is_single else "  <-- PROBLEM"
    print(f"{repr(label):<8} {str(ids):<25} {'yes' if is_single else 'NO'}{flag}")
    if not is_single:
        problems.append(label)

print("=" * 55)

# Decode each single token back as a sanity check
print("\nRound-trip decode (encode then decode, for single-token strings):")
for label, text in CANDIDATES.items():
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        decoded = tokenizer.decode(ids)
        match = "OK" if decoded == text else f"MISMATCH (got {repr(decoded)})"
        print(f"  {repr(text)} -> token {ids[0]} -> {repr(decoded)}  [{match}]")

# Summary
print()
if problems:
    print(f"FAILED: the following strings are NOT single tokens: {problems}")
    print("Logit extraction will be unreliable. Debug before proceeding.")
else:
    print("All 8 strings are single tokens. Safe to hardcode IDs.")

# Print the four space-prefixed IDs in copy-pasteable form for hardcoding
spaced = {label: tokenizer.encode(text, add_special_tokens=False)[0]
          for label, text in CANDIDATES.items()
          if text.startswith(" ")
          and len(tokenizer.encode(text, add_special_tokens=False)) == 1}

if len(spaced) == 4:
    print("\nHardcode these in extract_logits.py:")
    print(f"  TOKEN_A = {spaced[' A']}")
    print(f"  TOKEN_B = {spaced[' B']}")
    print(f"  TOKEN_1 = {spaced[' 1']}")
    print(f"  TOKEN_2 = {spaced[' 2']}")
