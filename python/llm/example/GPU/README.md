# BigDL-LLM Examples on Intel GPU

This folder contains examples of running BigDL-LLM on Intel GPU:

- [HF-Transformers-AutoModels](HF-Transformers-AutoModels): running any ***Hugging Face Transformers*** model on BigDL-LLM (using the standard AutoModel APIs)
- [QLoRA-FineTuning](QLoRA-FineTuning): running ***QLoRA finetuning*** using BigDL-LLM on Intel GPUs
- [vLLM-Serving](vLLM-Serving): running ***vLLM*** serving framework on intel GPUs (with BigDL-LLM low-bit optimized models)
- [Deepspeed-AutoTP](Deepspeed-AutoTP): running distributed inference using ***DeepSpeed AutoTP*** (with BigDL-LLM low-bit optimized models) on Intel GPUs
- [PyTorch-Models](PyTorch-Models): running any PyTorch model on BigDL-LLM (with "one-line code change")