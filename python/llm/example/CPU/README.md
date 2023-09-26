# BigDL-LLM Optimization for Large Language Model on Intel CPUs

We provide detailed examples to help you run popular open-source models using BigDL-LLM on Intel CPUs. Please refer to the appropriate guide based on different BigDL-LLM key feature:

## Pytorch API

In general, you just need one-line `optimize_model` to easily optimize any loaded PyTorch model, regardless of the library or API you are using. See the complete example [here](PyTorch-Models), including:

- [Model](PyTorch-Models/Model): examples to run some popular models with INT4 optimizations.
- [More-Data-Types](PyTorch-Models/More-Data-Types): examples of how to apply other low bit optimizations.
- [Save-Load](PyTorch-Models/Save-Load): examples about how to save and load optimized model.

## Transformers-style API

#### Huggingface transformers Format

Many popular open-source PyTorch large language models can be loaded using the Huggingface Transformers API (such as AutoModel, AutoModelForCasualLM, etc). For such models, BigDL-LLM also provides a set of APIs to support them. See the complete example [here](HF-Transformers-AutoModels), including:

- [Model](HF-Transformers-AutoModels/Model): examples to run some popular models with INT4 optimizations.
- [More-Data-Types](HF-Transformers-AutoModels/More-Data-Types): examples of how to apply other low bit optimizations.
- [Save-Load](HF-Transformers-AutoModels/Save-Load): examples about how to save and load optimized model.

#### Native Format

Especially, for `llama`/`bloom`/`gptneox`/`starcoder`/`chatglm` model families, BigDL-LLM supports native INT4 format on Intel CPUs to realize maximum performance and examples can be found [here](Native-Models).


## LangChain API

BigDL-LLM also provides LangChain integrations (i.e. LLM wrappers and embeddings) and complete examples can be found [here](LangChain).
