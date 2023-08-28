# BigDL-LLM Transformers INT4 Optimization for Large Language Model on Intel® Arc™ A-Series Graphics
You can use BigDL-LLM to run almost every Huggingface Transformer models with INT4 optimizations on your laptops with Intel® Arc™ A-Series Graphics. This directory contains example scripts to help you quickly get started using BigDL-LLM to run some popular open-source models in the community. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Recommended Requirements
To apply Intel® Arc™ A-Series Graphics acceleration, there’re several steps for tools installation and environment preparation.

Step 1, only Linux system is supported now, Ubuntu 22.04 is prefered.

Step 2, please refer to our [drive installation](https://dgpu-docs.intel.com/installation-guides/index.html#intel-arc-gpus) for general purpose GPU capabilities.

Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
> **Note**: IPEX 2.0.110+xpu requires Intel® oneAPI Base Toolkit's version >= 2023.2.0.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux:
```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
