# Colab session setup

Paste these cells at the top of every new Colab session. They mount Google
Drive and route the HuggingFace cache there so the 16.8 GB Qwen2-Audio model
only downloads once (first session) and loads from Drive thereafter.

---

## Cell 1 — mount Drive + set HF cache (run FIRST, before any other import)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
HF_CACHE = '/content/drive/MyDrive/hf_cache'
os.makedirs(HF_CACHE, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
os.environ['HF_HUB_CACHE'] = HF_CACHE
print(f"HF cache → {HF_CACHE}")
```

**Why this must run first:** `transformers` and `datasets` read `HF_HOME` at
import time. If you import them before setting the env var, they latch onto
the default `~/.cache/huggingface` path and ignore your override.

---

## Cell 2 — install deps + confirm GPU

```python
!pip install transformers bitsandbytes accelerate datasets soundfile -q
!nvidia-smi
```

Confirm a GPU with ≥16 GB is attached (T4 is typical).

---

## Cell 3 — upload project scripts

Drag the `scripts/` folder from your local machine into the Colab Files
panel (left sidebar). Or recreate the structure:

```python
!mkdir -p scripts results
# then upload scripts/extract_logits.py via the Files panel
```

---

## Cell 4 — run extraction

The `!` form launches a subprocess that inherits the env vars set in cell 1,
so the script sees the Drive cache path:

```python
!python scripts/extract_logits.py --format both --resume
```

`--resume` skips items already saved in `results/logits_primary_*.jsonl`.

---

## Cell 5 — sanity-check output

```python
!wc -l results/logits_primary_A.jsonl results/logits_primary_B.jsonl
!head -1 results/logits_primary_A.jsonl
```

Both files should have 218 lines when the full run completes.

---

## Cell 6 — download results

Right-click each `.jsonl` in the Files panel → Download. Move into
`/home/piyush/Desktop/IS/results/` on your local machine.

---

## Notes

- **Session 1:** model downloads to Drive (~56 min at HF's unauthenticated
  rate limit). One-time cost.
- **Future sessions:** model loads from Drive (~2–5 min, no network).
- **If a download is ever interrupted mid-session:** the next `from_pretrained`
  call resumes the partial file. No manual cleanup needed.
- **Drive subfolder layout** (auto-created by HuggingFace):
  ```
  /content/drive/MyDrive/hf_cache/
    hub/
      models--Qwen--Qwen2-Audio-7B-Instruct/
      datasets--slprl--StressTest/
  ```
