# Self-Speculative Decoding for Large Language Model BF16 Inference using BigDL-LLM on Intel CPUs
You can use BigDL-LLM to run BF16 inference for any Huggingface Transformer model with ***self-speculative decoding*** on Intel CPUs. This directory contains example scripts to help you quickly get started to run some popular open-source models using self-speculative decoding. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Verified Hardware Platforms

- Intel Xeon SPR server

## Recommended Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#system-support) for more information. Make sure you have installed `bigdl-llm` before:

```bash
pip install --pre --upgrade bigdl-llm[all]
```

Moreover, install IPEX 2.1.0, which can be done through `pip install intel_extension_for_pytorch==2.1.0`.
