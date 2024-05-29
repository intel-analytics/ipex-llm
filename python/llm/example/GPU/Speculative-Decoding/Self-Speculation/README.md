# Self-Speculative Decoding for Large Language Model FP16 Inference using IPEX-LLM on Intel GPUs
You can use IPEX-LLM to run FP16 inference for any Huggingface Transformer model with ***self-speculative decoding*** on Intel GPUs. This directory contains example scripts to help you quickly get started to run some popular open-source models using self-speculative decoding. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Verified Hardware Platforms

- Intel Data Center GPU Max Series

## Recommended Requirements
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation. See the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more details.

Step 1, only Linux system is supported now, Ubuntu 22.04 is prefered.

Step 2, please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
> **Note**: IPEX 2.1.10+xpu requires Intel GPU Driver version >= stable_775_20_20231219.

Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
> **Note**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0.

## Best Known Configuration on Linux

For optimal performance on Intel Data Center GPU Max Series, it is recommended to set several environment variables.
```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
