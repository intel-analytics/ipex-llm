# BigDL-LLM Optimization for Large Language Model on Intel GPUs

We provide detailed examples to help you run popular open-source models using BigDL-LLM on Intel GPUs. Please refer to the appropriate guide based on different BigDL-LLM key feature:

## PyTorch API

In general, you just need one-line `optimize_model` to easily optimize any loaded PyTorch model, regardless of the library or API you are using. See the complete example in [PyTorch-Models Folder](PyTorch-Models).

## Transformers-style API

Many popular open-source PyTorch large language models can be loaded using the Huggingface Transformers API (such as AutoModel, AutoModelForCasualLM, etc). For such models, BigDL-LLM also provides a set of APIs to support them. See the complete example in [HF-Transformers-AutoModels Folder](HF-Transformers-AutoModels).

## QLoRA FineTuning

BigDL-LLM also supports finetuning with 4bit optimizations on Intel GPUs and complete example can be found in [QLoRA-FineTuning Folder](QLoRA-FineTuning).
