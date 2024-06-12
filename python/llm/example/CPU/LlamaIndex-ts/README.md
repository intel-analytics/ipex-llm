# LlamaIndex-ts

This folder contains examples showcasing how to use [**LlamaIndex-ts**](https://ts.llamaindex.ai/guides/agents/setup) with `ipex-llm`.
> [**LlamaIndex-ts**](https://ts.llamaindex.ai/guides/agents/setup) helps you unlock domain-specific data and then build powerful applications with it.

## Setting up Dependencies 

### Install LlamaIndex-ts
Following [this page](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/LlamaIndex/README.md) to create an environment for llamaindex.
Then run the commands below
```
npm install llamaindex
npm install dotenv
```

### Install Ollama
* Origin Ollama: refer to [this page](https://github.com/ollama/ollama) to install Ollama on CPU. 
* Ollama enabled by IPEX-LLM: you can use commands below to install and run ollama. 
    ```
    pip install --pre --upgrade ipex-llm[cpp]
    init-ollama
    ```
