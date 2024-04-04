# Running HuggingFace `transformers` model using IPEX-LLM on Intel GPU

This folder contains examples of running any HuggingFace `transformers` model on IPEX-LLM (using the standard AutoModel APIs):

- [Model](Model): examples of running HuggingFace transformers models (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) using INT4 optimizations
- [More-Data-Types](More-Data-Types): examples of applying other low bit optimizations (FP8/INT8/FP4, etc.)
- [Save-Load](Save-Load): examples of saving and loading low-bit models
- [Advanced-Quantizations](Advanced-Quantizations): examples of loading GGUF/AWQ/GPTQ models
