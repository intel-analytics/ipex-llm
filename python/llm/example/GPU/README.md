# IPEX-LLM Examples on Intel GPU

This folder contains examples of running IPEX-LLM on Intel GPU:

- [Applications](Applications): running LLM applications (such as autogen) on IPEX-LLM
- [HuggingFace](HuggingFace): running ***HuggingFace*** models on IPEX-LLM (using the standard AutoModel APIs), including language models and multimodal models.
- [LLM-Finetuning](LLM-Finetuning): running ***finetuning*** (such as LoRA, QLoRA, QA-LoRA, etc) using IPEX-LLM on Intel GPUs
- [vLLM-Serving](vLLM-Serving): running ***vLLM*** serving framework on intel GPUs (with IPEX-LLM low-bit optimized models)
- [Deepspeed-AutoTP](Deepspeed-AutoTP): running distributed inference using ***DeepSpeed AutoTP*** (with IPEX-LLM low-bit optimized models) on Intel GPUs
- [Deepspeed-AutoTP-FastAPI](Deepspeed-AutoTP-FastAPI): running distributed inference using ***DeepSpeed AutoTP*** and start serving with ***FastAPI***(with IPEX-LLM low-bit optimized models) on Intel GPUs
- [Pipeline-Parallel-Inference](Pipeline-Parallel-Inference): running IPEX-LLM optimized low-bit model vertically partitioned on multiple Intel GPUs
- [Pipeline-Parallel-Serving](Pipeline-Parallel-Serving): running IPEX-LLM serving with **FastAPI** on multiple Intel GPUs in pipeline parallel fasion
- [Lightweight-Serving](Lightweight-Serving): running IPEX-LLM serving with **FastAPI** on one Intel GPU In a lightweight way
- [LangChain](LangChain): running ***LangChain*** applications on IPEX-LLM
- [PyTorch-Models](PyTorch-Models): running any PyTorch model on IPEX-LLM (with "one-line code change")
- [Speculative-Decoding](Speculative-Decoding): running any ***Hugging Face Transformers*** model with ***self-speculative decoding*** on Intel GPUs
- [ModelScope-Models](ModelScope-Models): running ***ModelScope*** model with IPEX-LLM on Intel GPUs
- [Long-Context](Long-Context): running **long-context** generation with IPEX-LLM on Intel Arc™ A770 Graphics.


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
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation. See the GPU installation guide on [Linux](../../../../docs/mddocs/Quickstart/install_linux_gpu.md) or [Windows](../../../../docs/mddocs/Quickstart/install_windows_gpu.md) for mode details.