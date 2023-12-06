# BigDL-LLM Examples on Intel GPU

This folder contains examples of running BigDL-LLM on Intel GPU:

- [HF-Transformers-AutoModels](HF-Transformers-AutoModels): running any ***Hugging Face Transformers*** model on BigDL-LLM (using the standard AutoModel APIs)
- [QLoRA-FineTuning](QLoRA-FineTuning): running ***QLoRA finetuning*** using BigDL-LLM on Intel GPUs
- [vLLM-Serving](vLLM-Serving): running ***vLLM*** serving framework on intel GPUs (with BigDL-LLM low-bit optimized models)
- [Deepspeed-AutoTP](Deepspeed-AutoTP): running distributed inference using ***DeepSpeed AutoTP*** (with BigDL-LLM low-bit optimized models) on Intel GPUs
- [PyTorch-Models](PyTorch-Models): running any PyTorch model on BigDL-LLM (with "one-line code change")


## System Support
**Hardware**:
- Intel Arc™ A-Series Graphics
- Intel Data Center GPU Flex Series
- Intel Data Center GPU Max Series

**Operating System**:
- Ubuntu 20.04 or later (Ubuntu 22.04 is preferred)

## Requirements
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation.

Step 1, please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
> **Note**: IPEX 2.0.110+xpu requires Intel GPU Driver version is [Stable 647.21](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html).

Step 2, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
> **Note**: IPEX 2.0.110+xpu requires Intel® oneAPI Base Toolkit's version >= 2023.2.0.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux:
```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
