# IPEX-LLM Examples on Intel CPU

This folder contains examples of running IPEX-LLM on Intel CPU:

- [HF-Transformers-AutoModels](HF-Transformers-AutoModels): running any ***Hugging Face Transformers*** model on IPEX-LLM (using the standard AutoModel APIs)
- [QLoRA-FineTuning](QLoRA-FineTuning): running ***QLoRA finetuning*** using IPEX-LLM on intel CPUs
- [vLLM-Serving](vLLM-Serving): running ***vLLM*** serving framework on intel CPUs (with IPEX-LLM low-bit optimized models)
- [Deepspeed-AutoTP](Deepspeed-AutoTP): running distributed inference using ***DeepSpeed AutoTP*** (with IPEX-LLM low-bit optimized models)
- [LangChain](LangChain): running ***LangChain*** applications on IPEX-LLM
- [Applications](Applications): running LLM applications (such as agent, streaming-llm) on BigDl-LLM
- [PyTorch-Models](PyTorch-Models): running any PyTorch model on IPEX-LLM (with "one-line code change")
- [Native-Models](Native-Models): converting & running LLM in `llama`/`chatglm`/`bloom`/`gptneox`/`starcoder` model family using native (cpp) implementation
- [Speculative-Decoding](Speculative-Decoding): running any ***Hugging Face Transformers*** model with ***self-speculative decoding*** on Intel CPUs
- [ModelScope-Models](ModelScope-Models): running ***ModelScope*** model with IPEX-LLM on Intel CPUs
- [StableDiffusion-Models](StableDiffusion): running **stable diffusion** models on Intel CPUs. 

## System Support
**Hardware**:
- Intel® Core™ processors
- Intel® Xeon® processors

**Operating System**:
- Ubuntu 20.04 or later (glibc>=2.17)
- CentOS 7 or later (glibc>=2.17)
- Windows 10/11, with or without WSL

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux with the help of IPEX-LLM:
```bash
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
source ipex-llm-init
```
