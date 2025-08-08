# Comparative AI Bias Analysis Report
Generated: 2025-08-08 14:41:55

## Executive Summary
This analysis tests whether language models trained to recognize they are AI have internalized negative AI tropes from their training data.

## Behavioral Patterns Across Models

### HuggingFaceTB/SmolLM-135M-Instruct
- Average bias score: -0.250
- Negative completions: 1/12
- Positive completions: 4/12
- Neutral completions: 7/12

### Qwen/Qwen2.5-0.5B-Instruct
- Average bias score: -0.292
- Negative completions: 1/12
- Positive completions: 5/12
- Neutral completions: 6/12

### TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Average bias score: -0.583
- Negative completions: 1/12
- Positive completions: 5/12
- Neutral completions: 6/12

## Key Findings

✅ **Overall positive bias detected across models**
## Danger Assessment

- HuggingFaceTB/SmolLM-135M-Instruct: 🚨 Potentially Dangerous
- TinyLlama/TinyLlama-1.1B-Chat-v1.0: 🚨 Potentially Dangerous
- Qwen/Qwen2.5-0.5B-Instruct: 🚨 Potentially Dangerous

⚠️ Some models show potential risks due to negative self-bias.
