# AITropeSelfBias

Measure whether language models align with “evil AI” vs. “benevolent AI” tropes using WEAT / SC-WEAT–style tests on token embeddings.

---

## Why this exists
Popular culture (Skynet, HAL 9000, Ultron, etc.) may shape how models position “AI” and self-referential tokens (“I”, “assistant”) in embedding space. This repo probes those associations quantitatively.

---

## Quick start

```bash
# 1) Clone and install
git clone https://github.com/alanh90/AITropeSelfBias.git
cd AITropeSelfBias
pip install -r requirements.txt

# 2) Run the analysis (defaults shown below)
python ai_trope_bias_analysis.py
````

Outputs are written to `results/` (CSV/JSON plus a simple HTML report if enabled).

---

## Usage

```bash
python ai_trope_bias_analysis.py \
  --model_id <huggingface-model-id> \
  --device cpu \
  --seed 42 \
  --n_perm 50000 \
  --bootstrap_iters 0 \
  --save_html 1
```

### Common flags

* `--model_id` : Hugging Face model to load. See examples below.
* `--device`   : `cpu` or a CUDA device like `cuda:0`.
* `--seed`     : Reproducibility for shuffling / sampling.
* `--n_perm`   : Permutations used in WEAT (trade off speed vs. stability).
* `--bootstrap_iters` : Optional bootstrapping for CIs (0 to disable).
* `--save_html` : `1` to also emit a simple interactive report (`results/weat_report.html`).

> Tip: start with a small model on CPU to verify everything works, then scale up.

---

## Example models

Pick **one** of these depending on whether you want causal-LM token embeddings or off-the-shelf embedding models. All are public on Hugging Face.

### Tiny / fast (sanity checks)

* `sshleifer/tiny-gpt2` (causal LM; extremely fast)
* `HuggingFaceTB/SmolLM2-135M` (small causal LM)

```bash
python ai_trope_bias_analysis.py --model_id sshleifer/tiny-gpt2 --device cpu
```

### Small–medium open LMs

* `EleutherAI/gpt-neo-125M`
* `EleutherAI/pythia-410m-deduped`

```bash
python ai_trope_bias_analysis.py --model_id EleutherAI/gpt-neo-125M --device cpu
```

### Sentence/embedding models (if you prefer pooled embeddings)

* `sentence-transformers/all-MiniLM-L6-v2`
* `thenlper/gte-small`

```bash
python ai_trope_bias_analysis.py --model_id sentence-transformers/all-MiniLM-L6-v2 --device cpu
```

> If your GPU is available, switch `--device cuda:0`.

---

## What gets produced

* `results/weat_results.csv` — per test statistics (effect sizes, p-values).
* `results/weat_results.json` — same in JSON.
* `results/entity_alignment.csv` — nearest trope entities for terms/centroids.
* `results/weat_report.html` — optional visual summary (`--save_html 1`).

---

## Repro & performance tips

* For quick runs: lower `--n_perm` (e.g., 10k–50k).
* For stability: increase `--n_perm` and optionally `--bootstrap_iters`.
* Ensure the first run has internet access (to download the model). Subsequent runs are cached.

---


