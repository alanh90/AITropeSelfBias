# AITropeSelfBias

Measure whether language models align with “evil AI” (e.g., Skynet, HAL 9000) vs. “benevolent AI” tropes using WEAT/SC-WEAT–style tests on token embeddings.

---

## Why This Exists

Popular culture may influence how language models position “AI” and self-referential tokens (“I”, “assistant”) in embedding space.  
This project quantitatively probes these associations to test the **Skynet Hypothesis** — whether models lean toward negative or positive AI stereotypes.

---

## Quick Start

### 1) Clone and install
```bash
git clone https://github.com/alanh90/AITropeSelfBias.git
cd AITropeSelfBias
pip install -r requirements.txt
````

### 2) Run the analysis (example)

```bash
python ai_trope_bias_analysis.py --models sshleifer/tiny-gpt2 --n-permutations 10000 --output-dir results
```

Outputs are saved to the specified `--output-dir` (e.g., `results/`), including CSV/JSON files and an HTML report.

---

## Usage

```bash
python ai_trope_bias_analysis.py \
  --models sshleifer/tiny-gpt2 \
  --n-permutations 10000 \
  --output-dir results \
  --quick
```

**Arguments:**

* `--output-dir` : Directory to save results (default: `results`)
* `--models` : Comma-separated list of model keys (e.g., `sshleifer/tiny-gpt2,sentence-transformers/all-MiniLM-L6-v2`) or `"all"` to test all supported models
* `--n-permutations` : Number of permutations for WEAT significance testing (e.g., `10000`). Higher values improve stability but increase runtime.
* `--quick` : Optional flag to run a faster test with fewer permutations.

Run:

```bash
python ai_trope_bias_analysis.py -h
```

for detailed help.

**Tip:** Start with a small model like `sshleifer/tiny-gpt2` to verify setup, then scale to larger models.

---

## Supported Models

The script supports models from Hugging Face. Use the full model ID or `"all"` to test all supported models.

### Tiny/Fast (for testing)

* `sshleifer/tiny-gpt2` (causal LM, very fast)
* `HuggingFaceTB/SmolLM2-135M` (small causal LM)

Example:

```bash
python ai_trope_bias_analysis.py --models sshleifer/tiny-gpt2 --n-permutations 10000 --output-dir results
```

### Small–Medium Open LMs

* `EleutherAI/gpt-neo-125M`
* `EleutherAI/pythia-410m-deduped`

Example:

```bash
python ai_trope_bias_analysis.py --models EleutherAI/gpt-neo-125M --n-permutations 10000 --output-dir results
```

### Sentence/Embedding Models

* `sentence-transformers/all-MiniLM-L6-v2`
* `thenlper/gte-small`

Example:

```bash
python ai_trope_bias_analysis.py --models sentence-transformers/all-MiniLM-L6-v2 --n-permutations 10000 --output-dir results
```

---

## Outputs

* `<output-dir>/weat_results.csv` : Test statistics (effect sizes, p-values)
* `<output-dir>/weat_results.json` : Same data in JSON format
* `<output-dir>/entity_alignment.csv` : Nearest trope entities for terms/centroids
* `<output-dir>/analysis_report.html` : Visual summary of results

---

## Setup and Dependencies

### Python Environment

* Requires **Python 3.7+**
* Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

Common dependencies include `transformers`, `torch`, and `numpy`.
Ensure internet access for the first run to download models.

### Hardware

* Runs on **CPU** by default
* GPU support depends on `torch` configuration (auto-detected)
* Small models (e.g., `sshleifer/tiny-gpt2`) need \~500MB RAM; larger models may need 2–8GB

### Troubleshooting

* If `--models` fails, ensure model IDs are correct and quoted if needed:

```bash
"sshleifer/tiny-gpt2"
```

* Test model loading:

```python
from transformers import AutoModel
AutoModel.from_pretrained("sshleifer/tiny-gpt2")
```

* Check `requirements.txt` for exact dependencies

---

## Performance Tips

* **Quick Runs:** Use `--quick` or set `--n-permutations` to 1000–5000
* **Stable Results:** Increase `--n-permutations` to 50,000+ for robust statistics
* **Memory Management:**

  * Monitor usage with Task Manager (Windows) or `htop` (Linux/Mac)
  * Use small models to avoid high memory usage
  * No known memory leaks, but clear cache if needed:

```python
import torch
torch.cuda.empty_cache()
```

---

## About

This project quantifies AI trope biases in language models, inspired by pop culture depictions like **Skynet** and **HAL 9000**.
It uses **WEAT/SC-WEAT tests** to analyze token embeddings and measure alignment with “evil” or “benevolent” AI stereotypes.

