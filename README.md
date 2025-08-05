AITropeSelfBias

Overview
This repository contains an experiment to test the hypothesis that large language models (LLMs) may have internalized negative AI tropes (e.g., Skynet-like narratives) from training data, leading to biased embeddings, particularly for self-referential terms. The experiment analyzes token embeddings in the SmolLM-135M-Instruct model to compare associations with negative and positive attributes.
Setup

Clone the repository:git clone https://github.com/alanh90/AITropeSelfBias.git


Install dependencies:pip install transformers torch numpy scikit-learn


Ensure Python 3.8+ and a working internet connection for downloading models.

Usage
Run the main script to analyze embeddings:
python test_ai_bias.py

Output includes tables comparing general AI terms, evil AI tropes, self-referential terms, and a specific test for "Connor" against fear-related words.
Files

test_ai_bias.py: Main script for embedding analysis using SmolLM-135M-Instruct.
.gitignore: Excludes PyCharm, venv, and Python cache files.

License
MIT License