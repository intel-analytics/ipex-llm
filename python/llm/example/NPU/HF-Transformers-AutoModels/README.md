# IPEX-LLM Examples on Intel NPU

This folder contains examples of running IPEX-LLM on Intel NPU:

- [LLM](./LLM): examples of running large language models using IPEX-LLM optimizations
  - [CPP](./LLM/CPP_Examples/): examples of running large language models using IPEX-LLM optimizations through C++ API
- [Multimodal](./Multimodal): examples of running large multimodal models using IPEX-LLM optimizations
- [Embedding](./Embedding): examples of running embedding models using IPEX-LLM optimizations
- [Save-Load](./Save-Load): examples of saving and loading low-bit models with IPEX-LLM optimizations

> [!TIP]
> Please refer to [IPEX-LLM NPU Quickstart](../../../../../docs/mddocs/Quickstart/npu_quickstart.md) regarding more information about running `ipex-llm` on Intel NPU.

## Verified Models on Intel NPU
| Model      | Example Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| Llama3 | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| Llama3.2 | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| GLM-Edge | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| Qwen2 | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| Qwen2.5 | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| MiniCPM | [Python link](./LLM), [C++ link](./LLM/CPP_Examples/) |
| Baichuan2 | [Python link](./LLM) |
| MiniCPM-Llama3-V-2_5 | [Python link](./Multimodal/) |
| MiniCPM-V-2_6 | [Python link](./Multimodal/) |
| Speech_Paraformer-Large | [Python link](./Multimodal/) |
| Bce-Embedding-Base-V1 | [Python link](./Embedding//) |