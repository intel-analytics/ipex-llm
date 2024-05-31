# Self-Speculative Decoding for Large Language Model BF16 Inference using IPEX-LLM on Intel CPUs
You can use IPEX-LLM to run BF16 inference for any Huggingface Transformer model with ***self-speculative decoding*** on Intel CPUs. This directory contains example scripts to help you quickly get started to run some popular open-source models using self-speculative decoding. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Verified Hardware Platforms

- Intel Xeon SPR server

## Recommended Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../README.md#system-support) for more information. Make sure you have installed `ipex-llm` before:

```bash
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

Moreover, install IPEX 2.1.0, which can be done through `pip install intel_extension_for_pytorch==2.1.0`.
