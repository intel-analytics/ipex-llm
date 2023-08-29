# BigDL-LLM Transformers INT4 Optimization for Large Language Model on Intel GPUs
You can use BigDL-LLM to run almost every Huggingface Transformer models with INT4 optimizations on your laptops with Intel GPUs. This directory contains example scripts to help you quickly get started using BigDL-LLM to run some popular open-source models in the community. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Verified Hardware Platforms

- Intel Arc™ A-Series Graphics
- Intel Data Center GPU Flex Series

## Recommended Requirements
To apply Intel® GPU acceleration, there’re several steps for tools installation and environment preparation.

Step 1, only Linux system is supported now, Ubuntu 22.04 is prefered.

Step 2, please refer to our [drive installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.

Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux:
```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
