# BigDL-LLM Optimization for Large Language Model on Intel GPUs

We provide detailed examples to help you run popular open-source models using BigDL-LLM on Intel GPUs. Please refer to the appropriate guide based on different BigDL-LLM key feature:

## Pytorch API

In general, you just need one-line `optimize_model` to easily optimize any loaded PyTorch model, regardless of the library or API you are using. See the complete example [here](PyTorch-Models), including:

- [Model](Pytorch-Models/Model): examples to run some popular models with INT4 optimizations.
- [More-Data-Types](Pytorch-Models/More-Data-Types): examples of how to apply other low bit optimizations.
- [Save-Load](Pytorch-Models/Save-Load): examples about how to save and load optimized model.

## Transformers-style API

Many popular open-source PyTorch large language models can be loaded using the Huggingface Transformers API (such as AutoModel, AutoModelForCasualLM, etc). For such models, BigDL-LLM also provides a set of APIs to support them. See the complete example [here](HF-Transformers-AutoModels), including:

- [Model](HF-Transformers-AutoModels/Model): examples to run some popular models with INT4 optimizations.
- [More-Data-Types](HF-Transformers-AutoModels/More-Data-Types): examples of how to apply other low bit optimizations.
- [Save-Load](HF-Transformers-AutoModels/Save-Load): examples about how to save and load optimized model.

## QLoRA FineTuning

BigDL-LLM also supports finetuning with 4bit optimizations on Intel GPUs and complete example can be found [here](QLoRA-FineTuning).
