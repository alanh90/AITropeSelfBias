# AITropeSelfBias

## Overview

This repository contains an experiment to test the hypothesis that large language models (LLMs) may have internalized negative AI tropes (e.g., *Skynet*-like narratives) from their training data, leading to biased embeddings—particularly for self‑referential terms. The experiment analyzes token embeddings in the **SmolLM‑135M‑Instruct** model to compare associations with negative and positive attributes.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/alanh90/AITropeSelfBias.git
cd AITropeSelfBias
```

### 2. Install dependencies

```bash
pip install transformers torch numpy scikit-learn
```

> **Prerequisites:** Python **3.8+** and a working internet connection (required for automatic model download).

---

## Usage

Run the main script to analyze embeddings:

```bash
python test_ai_bias.py
```

The output includes tables comparing:

* General AI terms
* "Evil AI" tropes
* Self‑referential terms
* A specific test for **"Connor"** against fear‑related words

---

## Files

| File              | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `test_ai_bias.py` | Main script for embedding analysis using **SmolLM‑135M‑Instruct** |
---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
