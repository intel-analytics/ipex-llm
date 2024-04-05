# Running Hugging Face Transformers model using IPEX-LLM on Intel CPU

This folder contains examples of running any HuggingFace `transformers` model on IPEX-LLM (using the standard AutoModel APIs):

- [Model](Model): examples of running HuggingFace `transformers` models (e.g., LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) using INT4 optimizations
- [More-Data-Types](More-Data-Types): examples of applying other low bit optimizations (INT8/INT5, etc.) on Intel CPU
- [Save-Load](Save-Load): examples of saving and loading low-bit models
- [Advanced-Quantizations](Advanced-Quantizations): examples of loading GGUF/AWQ/GPTQ models
