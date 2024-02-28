# BigDL-LLM Examples on Intel GPU

This folder contains examples of running BigDL-LLM on Intel GPU:

- [Applications](Applications): running LLM applications (such as autogen) on BigDL-LLM
- [HF-Transformers-AutoModels](HF-Transformers-AutoModels): running any ***Hugging Face Transformers*** model on BigDL-LLM (using the standard AutoModel APIs)
- [LLM-Finetuning](LLM-Finetuning): running ***finetuning*** (such as LoRA, QLoRA, QA-LoRA, etc) using BigDL-LLM on Intel GPUs
- [vLLM-Serving](vLLM-Serving): running ***vLLM*** serving framework on intel GPUs (with BigDL-LLM low-bit optimized models)
- [Deepspeed-AutoTP](Deepspeed-AutoTP): running distributed inference using ***DeepSpeed AutoTP*** (with BigDL-LLM low-bit optimized models) on Intel GPUs
- [PyTorch-Models](PyTorch-Models): running any PyTorch model on BigDL-LLM (with "one-line code change")
- [Speculative-Decoding](Speculative-Decoding): running any ***Hugging Face Transformers*** model with ***self-speculative decoding*** on Intel GPUs
- [ModelScope-Models](ModelScope-Models): running ***ModelScope*** model with BigDL-LLM on Intel GPUs


## System Support
### 1. Linux: 
**Hardware**:
- Intel Arc™ A-Series Graphics
- Intel Data Center GPU Flex Series
- Intel Data Center GPU Max Series

**Operating System**:
- Ubuntu 20.04 or later (Ubuntu 22.04 is preferred)

### 2. Windows
**Hardware**:
- Intel iGPU and dGPU

**Operating System**:
- Windows 10/11, with or without WSL 

## Requirements
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation. See the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details.