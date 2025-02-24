#  💫 Intel® LLM Library for PyTorch* 
<p>
  < <a href='./README.md'>English</a> | <b>中文 ></b> 
</p>

## 最新更新 🔥 
- [2025/02] 新增 [Ollama Portable Zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) 在 Intel GPU 上直接**免安装运行 Ollama** (包括 [Windows](Quickstart/ollama_portable_zip_quickstart.zh-CN.md#windows用户指南) 和 [Linux](Quickstart/ollama_portable_zip_quickstart.zh-CN.md#linux用户指南))。
- [2025/02] 新增在 Intel Arc GPUs 上运行 [vLLM 0.6.6](DockerGuides/vllm_docker_quickstart.md) 的支持。
- [2025/01] 新增在 Intel Arc [B580](Quickstart/bmg_quickstart.md) GPU 上运行 `ipex-llm` 的指南。
- [2025/01] 新增在 Intel GPU 上运行 [Ollama 0.5.4](Quickstart/ollama_quickstart.zh-CN.md) 的支持。
- [2024/12] 增加了对 Intel Core Ultra [NPU](Quickstart/npu_quickstart.md)（包括 100H，200V 和 200K 系列）的 **Python** 和 **C++** 支持。

<details><summary>更多更新</summary>
<br/>

- [2024/11] 新增在 Intel Arc GPUs 上运行 [vLLM 0.6.2](DockerGuides/vllm_docker_quickstart.md) 的支持。
- [2024/07] 新增 Microsoft **GraphRAG** 的支持（使用运行在本地 Intel GPU 上的 LLM），详情参考[快速入门指南](Quickstart/graphrag_quickstart.md)。
- [2024/07] 全面增强了对多模态大模型的支持，包括 [StableDiffusion](../../python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion), [Phi-3-Vision](../../python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision), [Qwen-VL](../../python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl)，更多详情请点击[这里](../../python/llm/example/GPU/HuggingFace/Multimodal)。
- [2024/07] 新增 Intel GPU 上 **FP6** 的支持，详情参考[更多数据类型样例](../../python/llm/example/GPU/HuggingFace/More-Data-Types)。 
- [2024/06] 新增对 Intel Core Ultra 处理器中 **NPU** 的实验性支持，详情参考[相关示例](../../python/llm/example/NPU/HF-Transformers-AutoModels)。 
- [2024/06] 增加了对[流水线并行推理](../../python/llm/example/GPU/Pipeline-Parallel-Inference)的全面支持，使得用两块或更多 Intel GPU（如 Arc）上运行 LLM 变得更容易。
- [2024/06] 新增在 Intel GPU 上运行 **RAGFlow** 的支持，详情参考[快速入门指南](Quickstart/ragflow_quickstart.md)。
- [2024/05] 新增 **Axolotl** 的支持，可以在 Intel GPU 上进行LLM微调，详情参考[快速入门指南](Quickstart/axolotl_quickstart.md)。
- [2024/05] 你可以使用 **Docker** [images](#docker) 很容易地运行 `ipex-llm` 推理、服务和微调。
- [2024/05] 你能够在 Windows 上仅使用 "*[one command](Quickstart/install_windows_gpu.zh-CN.md#安装-ipex-llm)*" 来安装 `ipex-llm`。
- [2024/04] 你现在可以在 Intel GPU 上使用 `ipex-llm` 运行 **Open WebUI** ，详情参考[快速入门指南](Quickstart/open_webui_with_ollama_quickstart.md)。
- [2024/04] 你现在可以在 Intel GPU 上使用 `ipex-llm` 以及 `llama.cpp` 和 `ollama` 运行 **Llama 3** ，详情参考[快速入门指南](Quickstart/llama3_llamacpp_ollama_quickstart.md)。
- [2024/04] `ipex-llm` 现在在Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM/llama3) 和 [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) 上都支持 **Llama 3** 了。
- [2024/04] `ipex-llm` 现在提供 C++ 推理, 在 Intel GPU 上它可以用作运行 [llama.cpp](Quickstart/llama_cpp_quickstart.zh-CN.md) 和 [ollama](Quickstart/ollama_quickstart.zh-CN.md) 的加速后端。
- [2024/03] `bigdl-llm` 现已更名为 `ipex-llm` (请参阅[此处](Quickstart/bigdl_llm_migration.md)的迁移指南)，你可以在[这里](https://github.com/intel-analytics/bigdl-2.x)找到原始BigDL项目。
- [2024/02] `ipex-llm` 现在支持直接从 [ModelScope](../../python/llm/example/GPU/ModelScope-Models) ([魔搭](../../python/llm/example/CPU/ModelScope-Models)) loading 模型。
- [2024/02] `ipex-llm` 增加 **INT2** 的支持 (基于 llama.cpp [IQ2](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) 机制), 这使得在具有 16GB VRAM 的 Intel GPU 上运行大型 LLM（例如 Mixtral-8x7B）成为可能。
- [2024/02] 用户现在可以通过 [Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUI 使用 `ipex-llm`。
- [2024/02] `ipex-llm` 现在支持 *[Self-Speculative Decoding](Inference/Self_Speculative_Decoding.md)*，这使得在 Intel [GPU](../../python/llm/example/GPU/Speculative-Decoding) 和 [CPU](../../python/llm/example/CPU/Speculative-Decoding) 上为 FP16 和 BF16 推理带来 **~30% 加速** 。
- [2024/02] `ipex-llm` 现在支持在 Intel GPU 上进行各种 LLM 微调(包括 [LoRA](../../python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](../../python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) 和 [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora))。
- [2024/01] 使用 `ipex-llm` [QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA)，我们成功地在 8 个 Intel Max 1550 GPU 上使用 [Standford-Alpaca](../../python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora) 数据集分别对 LLaMA2-7B（**21 分钟内**）和 LLaMA2-70B（**3.14 小时内**）进行了微调，具体详情参阅[博客](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)。 
- [2023/12] `ipex-llm` 现在支持 [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora) (具体内容请参阅 *["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*).
- [2023/12] `ipex-llm` 现在在 Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM/mixtral) 和 [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral) 上均支持 [Mixtral-8x7B](../../python/llm/example/GPU/HuggingFace/LLM/mixtral)。
- [2023/12] `ipex-llm` 现在支持 [QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) (具体内容请参阅 *["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*). 
- [2023/12] `ipex-llm` 现在在 Intel ***GPU*** 上支持 [FP8 and FP4 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types)。
- [2023/11] 初步支持直接将 [GGUF](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF)，[AWQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ) 和 [GPTQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ) 模型加载到 `ipex-llm` 中。
- [2023/11] `ipex-llm` 现在在 Intel [GPU](../../python/llm/example/GPU/vLLM-Serving) 和 [CPU](../../python/llm/example/CPU/vLLM-Serving) 上都支持 [vLLM continuous batching](../../python/llm/example/GPU/vLLM-Serving) 。
- [2023/10] `ipex-llm` 现在在 Intel [GPU](../../python/llm/example/GPU/LLM-Finetuning/QLoRA) 和 [CPU](../../python/llm/example/CPU/QLoRA-FineTuning) 上均支持 [QLoRA finetuning](../../python/llm/example/GPU/LLM-Finetuning/QLoRA) 。
- [2023/10] `ipex-llm` 现在在 Intel GPU 和 CPU 上都支持 [FastChat serving](../../python/llm/src/ipex_llm/llm/serving) 。
- [2023/09] `ipex-llm` 现在支持 [Intel GPU](../../python/llm/example/GPU) (包括 iGPU, Arc, Flex 和 MAX)。
- [2023/09] `ipex-llm` [教程](https://github.com/intel-analytics/ipex-llm-tutorial) 已发布。
 
</details> 

## `ipex-llm` 快速入门

### 使用
- [Ollama Portable Zip](Quickstart/ollama_portable_zip_quickstart.zh-CN.md): 在 Intel GPU 上直接**免安装运行 Ollama**。
- [Arc B580](Quickstart/bmg_quickstart.md): 在 Intel Arc **B580** GPU 上运行 `ipex-llm`（包括 Ollama, llama.cpp, PyTorch, HuggingFace 等）
- [NPU](Quickstart/npu_quickstart.md): 在 Intel **NPU** 上运行 `ipex-llm`（支持 Python 和 C++）
- [llama.cpp](Quickstart/llama_cpp_quickstart.zh-CN.md): 在 Intel GPU 上运行 **llama.cpp** (*使用 `ipex-llm` 的 C++ 接口*) 
- [Ollama](Quickstart/ollama_quickstart.zh-CN.md): 在 Intel GPU 上运行 **ollama** (*使用 `ipex-llm` 的 C++ 接口*) 
- [PyTorch/HuggingFace](Quickstart/install_windows_gpu.zh-CN.md): 使用 [Windows](Quickstart/install_windows_gpu.zh-CN.md) 和 [Linux](Quickstart/install_linux_gpu.zh-CN.md) 在 Intel GPU 上运行 **PyTorch**、**HuggingFace**、**LangChain**、**LlamaIndex** 等 (*使用 `ipex-llm` 的 Python 接口*) 
- [vLLM](Quickstart/vLLM_quickstart.md): 在 Intel [GPU](DockerGuides/vllm_docker_quickstart.md) 和 [CPU](DockerGuides/vllm_cpu_docker_quickstart.md) 上使用 `ipex-llm` 运行 **vLLM** 
- [FastChat](Quickstart/fastchat_quickstart.md): 在 Intel GPU 和 CPU 上使用 `ipex-llm` 运行 **FastChat** 服务
- [Serving on multiple Intel GPUs](Quickstart/deepspeed_autotp_fastapi_quickstart.md): 利用 DeepSpeed AutoTP 和 FastAPI 在 **多个 Intel GPU** 上运行 `ipex-llm` 推理服务
- [Text-Generation-WebUI](Quickstart/webui_quickstart.md): 使用 `ipex-llm` 运行 `oobabooga` **WebUI** 
- [Axolotl](Quickstart/axolotl_quickstart.md): 使用 **Axolotl** 和 `ipex-llm` 进行 LLM 微调
- [Benchmarking](Quickstart/benchmark_quickstart.md):  在 Intel GPU 和 CPU 上运行**性能基准测试**（延迟和吞吐量）
 
### Docker
- [GPU Inference in C++](DockerGuides/docker_cpp_xpu_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `llama.cpp`, `ollama`等
- [GPU Inference in Python](DockerGuides/docker_pytorch_inference_gpu.md) : 在 Intel GPU 上使用 `ipex-llm` 运行 HuggingFace `transformers`, `LangChain`, `LlamaIndex`, `ModelScope`，等
- [vLLM on GPU](DockerGuides/vllm_docker_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `vLLM` 推理服务
- [vLLM on CPU](DockerGuides/vllm_cpu_docker_quickstart.md): 在 Intel CPU 上使用 `ipex-llm` 运行 `vLLM` 推理服务  
- [FastChat on GPU](DockerGuides/fastchat_docker_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `FastChat` 推理服务
- [VSCode on GPU](DockerGuides/docker_run_pytorch_inference_in_vscode.md): 在 Intel GPU 上使用 VSCode 开发并运行基于 Python 的 `ipex-llm` 应用

### 应用
- [GraphRAG](Quickstart/graphrag_quickstart.md): 基于 `ipex-llm` 使用本地 LLM 运行 Microsoft 的 `GraphRAG`
- [RAGFlow](Quickstart/ragflow_quickstart.md): 基于 `ipex-llm` 运行 `RAGFlow` (*一个开源的 RAG 引擎*)  
- [LangChain-Chatchat](Quickstart/chatchat_quickstart.md): 基于 `ipex-llm` 运行 `LangChain-Chatchat` (*使用 RAG pipline 的知识问答库*)
- [Coding copilot](Quickstart/continue_quickstart.md): 基于 `ipex-llm` 运行 `Continue` (VSCode 里的编码智能助手)
- [Open WebUI](Quickstart/open_webui_with_ollama_quickstart.md): 基于 `ipex-llm` 运行 `Open WebUI`
- [PrivateGPT](Quickstart/privateGPT_quickstart.md): 基于 `ipex-llm` 运行 `PrivateGPT` 与文档进行交互
- [Dify platform](Quickstart/dify_quickstart.md): 在`Dify`(*一款开源的大语言模型应用开发平台*) 里接入 `ipex-llm` 加速本地 LLM

### 安装
- [Windows GPU](Quickstart/install_windows_gpu.zh-CN.md): 在带有 Intel GPU 的 Windows 系统上安装 `ipex-llm` 
- [Linux GPU](Quickstart/install_linux_gpu.zh-CN.md): 在带有 Intel GPU 的Linux系统上安装 `ipex-llm` 
- *更多内容, 请参考[完整安装指南](Overview/install.md)*

### 代码示例
- #### 低比特推理
  - [INT4 inference](../../python/llm/example/GPU/HuggingFace/LLM): 在 Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM) 和 [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model) 上进行 **INT4** LLM 推理
  - [FP8/FP6/FP4 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types): 在 Intel [GPU](../../python/llm/example/GPU/HuggingFace/More-Data-Types) 上进行 **FP8**，**FP6** 和 **FP4** LLM 推理
  - [INT8 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types): 在 Intel [GPU](../../python/llm/example/GPU/HuggingFace/More-Data-Types) 和 [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types) 上进行 **INT8** LLM 推理 
  - [INT2 inference](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2): 在 Intel [GPU](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) 上进行 **INT2** LLM 推理 (基于 llama.cpp IQ2 机制) 
- #### FP16/BF16 推理
  - 在 Intel [GPU](../../python/llm/example/GPU/Speculative-Decoding) 上进行 **FP16** LLM 推理（并使用 [self-speculative decoding](Inference/Self_Speculative_Decoding.md) 优化）
  - 在 Intel [CPU](../../python/llm/example/CPU/Speculative-Decoding) 上进行 **BF16** LLM 推理（并使用 [self-speculative decoding](Inference/Self_Speculative_Decoding.md) 优化）
- #### 分布式推理
  - 在 Intel [GPU](../../python/llm/example/GPU/Pipeline-Parallel-Inference) 上进行 **流水线并行** 推理
  - 在 Intel [GPU](../../python/llm/example/GPU/Deepspeed-AutoTP) 上进行 **DeepSpeed AutoTP** 推理
- #### 保存和加载
  - [Low-bit models](../../python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load): 保存和加载 `ipex-llm` 低比特模型 (INT4/FP4/FP6/INT8/FP8/FP16/etc.)
  - [GGUF](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF): 直接将 GGUF 模型加载到 `ipex-llm` 中
  - [AWQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ): 直接将 AWQ 模型加载到 `ipex-llm` 中
  - [GPTQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ): 直接将 GPTQ 模型加载到 `ipex-llm` 中
- #### 微调
  - 在 Intel [GPU](../../python/llm/example/GPU/LLM-Finetuning) 进行 LLM 微调，包括 [LoRA](../../python/llm/example/GPU/LLM-Finetuning/LoRA)，[QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA)，[DPO](../../python/llm/example/GPU/LLM-Finetuning/DPO)，[QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) 和 [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora)
  - 在 Intel [CPU](../../python/llm/example/CPU/QLoRA-FineTuning) 进行 QLoRA 微调 
- #### 与社区库集成
  - [HuggingFace transformers](../../python/llm/example/GPU/HuggingFace)
  - [Standard PyTorch model](../../python/llm/example/GPU/PyTorch-Models)
  - [LangChain](../../python/llm/example/GPU/LangChain)
  - [LlamaIndex](../../python/llm/example/GPU/LlamaIndex)
  - [DeepSpeed-AutoTP](../../python/llm/example/GPU/Deepspeed-AutoTP)
  - [Axolotl](Quickstart/axolotl_quickstart.md)
  - [HuggingFace PEFT](../../python/llm/example/GPU/LLM-Finetuning/HF-PEFT)
  - [HuggingFace TRL](../../python/llm/example/GPU/LLM-Finetuning/DPO)
  - [AutoGen](../../python/llm/example/CPU/Applications/autogen)
  - [ModeScope](../../python/llm/example/GPU/ModelScope-Models)
- [教程](https://github.com/intel-analytics/ipex-llm-tutorial)

## API 文档
- [HuggingFace Transformers 兼容的 API (Auto Classes)](PythonAPI/transformers.md)
- [适用于任意 Pytorch 模型的 API](https://github.com/intel-analytics/ipex-llm/blob/main/PythonAPI/optimize.md)

## FAQ
- [常见问题解答](Overview/FAQ/faq.md)

## 模型验证
50+ 模型已经在 `ipex-llm` 上得到优化和验证，包括 *LLaMA/LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM2/ChatGLM3, Baichuan/Baichuan2, Qwen/Qwen-1.5, InternLM,* 更多模型请参看下表，
  
| 模型       | CPU 示例                                  | GPU 示例                                  | NPU 示例                                  |
|----------- |------------------------------------------|-------------------------------------------|-------------------------------------------|
| LLaMA  | [link1](../../python/llm/example/CPU/Native-Models), [link2](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](../../python/llm/example/GPU/HuggingFace/LLM/vicuna)|
| LLaMA 2    | [link1](../../python/llm/example/CPU/Native-Models), [link2](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/llama2)  | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) | [link](../../python/llm/example/GPU/HuggingFace/LLM/llama3)  | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3.1    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3.1) | [link](../../python/llm/example/GPU/HuggingFace/LLM/llama3.1)  |
| LLaMA 3.2    |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/llama3.2)  | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3.2-Vision    |  | [link](../../python/llm/example/GPU/PyTorch-Models/Model/llama3.2-vision/)  |
| ChatGLM    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |    | 
| ChatGLM2   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/chatglm2)   |
| ChatGLM3   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/chatglm3)   |
| GLM-4      | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm4)      | [link](../../python/llm/example/GPU/HuggingFace/LLM/glm4)       |
| GLM-4V     | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm-4v)    | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/glm-4v)     |
| GLM-Edge   |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/glm-edge) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM) |
| GLM-Edge-V   |  | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/glm-edge-v) |
| Mistral    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral)   | [link](../../python/llm/example/GPU/HuggingFace/LLM/mistral)    |
| Mixtral    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral)   | [link](../../python/llm/example/GPU/HuggingFace/LLM/mixtral)    |
| Falcon     | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon)    | [link](../../python/llm/example/GPU/HuggingFace/LLM/falcon)     |
| MPT        | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt)       | [link](../../python/llm/example/GPU/HuggingFace/LLM/mpt)        |
| Dolly-v1   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/dolly-v1)   | 
| Dolly-v2   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/dolly-v2)   | 
| Replit Code| [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit)    | [link](../../python/llm/example/GPU/HuggingFace/LLM/replit)     |
| RedPajama  | [link1](../../python/llm/example/CPU/Native-Models), [link2](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama) |    | 
| Phoenix    | [link1](../../python/llm/example/CPU/Native-Models), [link2](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix)   |    | 
| StarCoder  | [link1](../../python/llm/example/CPU/Native-Models), [link2](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder) | [link](../../python/llm/example/GPU/HuggingFace/LLM/starcoder) | 
| Baichuan   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/baichuan)   |
| Baichuan2  | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/baichuan2)  | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM) |
| InternLM   | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm)  | [link](../../python/llm/example/GPU/HuggingFace/LLM/internlm)   |
| InternVL2   |   | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/internvl2)   |
| Qwen       | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen)      | [link](../../python/llm/example/GPU/HuggingFace/LLM/qwen)       |
| Qwen1.5 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5) | [link](../../python/llm/example/GPU/HuggingFace/LLM/qwen1.5) |
| Qwen2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/qwen2) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| Qwen2.5 |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/qwen2.5) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| Qwen-VL    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen-vl)   | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl)    |
| Qwen2-VL    || [link](../../python/llm/example/GPU/PyTorch-Models/Model/qwen2-vl)    |
| Qwen2-Audio    |  | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/qwen2-audio)    |
| Aquila     | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila)    | [link](../../python/llm/example/GPU/HuggingFace/LLM/aquila)     |
| Aquila2     | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila2)    | [link](../../python/llm/example/GPU/HuggingFace/LLM/aquila2)     |
| MOSS       | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss)      |    | 
| Whisper    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper)   | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/whisper)    |
| Phi-1_5    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-1_5)   | [link](../../python/llm/example/GPU/HuggingFace/LLM/phi-1_5)    |
| Flan-t5    | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/flan-t5)   | [link](../../python/llm/example/GPU/HuggingFace/LLM/flan-t5)    |
| LLaVA      | [link](../../python/llm/example/CPU/PyTorch-Models/Model/llava)                 | [link](../../python/llm/example/GPU/PyTorch-Models/Model/llava)                  |
| CodeLlama  | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/codellama) | [link](../../python/llm/example/GPU/HuggingFace/LLM/codellama)  |
| Skywork      | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/skywork)                 |    |
| InternLM-XComposer  | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm-xcomposer)   |    |
| WizardCoder-Python | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/wizardcoder-python) | |
| CodeShell | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/codeshell) | |
| Fuyu      | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/fuyu) | |
| Distil-Whisper | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/distil-whisper) | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/distil-whisper) |
| Yi | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/yi) | [link](../../python/llm/example/GPU/HuggingFace/LLM/yi) |
| BlueLM | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/bluelm) | [link](../../python/llm/example/GPU/HuggingFace/LLM/bluelm) |
| Mamba | [link](../../python/llm/example/CPU/PyTorch-Models/Model/mamba) | [link](../../python/llm/example/GPU/PyTorch-Models/Model/mamba) |
| SOLAR | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/solar) | [link](../../python/llm/example/GPU/HuggingFace/LLM/solar) |
| Phixtral | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phixtral) | [link](../../python/llm/example/GPU/HuggingFace/LLM/phixtral) |
| InternLM2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/internlm2) |
| RWKV4 |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/rwkv4) |
| RWKV5 |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/rwkv5) |
| Bark | [link](../../python/llm/example/CPU/PyTorch-Models/Model/bark) | [link](../../python/llm/example/GPU/PyTorch-Models/Model/bark) |
| SpeechT5 |  | [link](../../python/llm/example/GPU/PyTorch-Models/Model/speech-t5) |
| DeepSeek-MoE | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek-moe) |  |
| Ziya-Coding-34B-v1.0 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/ziya) | |
| Phi-2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/phi-2) |
| Phi-3 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-3) | [link](../../python/llm/example/GPU/HuggingFace/LLM/phi-3) |
| Phi-3-vision | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-3-vision) | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision) |
| Yuan2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/yuan2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/yuan2) |
| Gemma | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/gemma) | [link](../../python/llm/example/GPU/HuggingFace/LLM/gemma) |
| Gemma2 |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/gemma2) |
| DeciLM-7B | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/deciLM-7b) | [link](../../python/llm/example/GPU/HuggingFace/LLM/deciLM-7b) |
| Deepseek | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek) | [link](../../python/llm/example/GPU/HuggingFace/LLM/deepseek) |
| StableLM | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/stablelm) | [link](../../python/llm/example/GPU/HuggingFace/LLM/stablelm) |
| CodeGemma | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/codegemma) | [link](../../python/llm/example/GPU/HuggingFace/LLM/codegemma) |
| Command-R/cohere | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/cohere) | [link](../../python/llm/example/GPU/HuggingFace/LLM/cohere) |
| CodeGeeX2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/codegeex2) | [link](../../python/llm/example/GPU/HuggingFace/LLM/codegeex2) |
| MiniCPM | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm) | [link](../../python/llm/example/GPU/HuggingFace/LLM/minicpm) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| MiniCPM3 |  | [link](../../python/llm/example/GPU/HuggingFace/LLM/minicpm3) |
| MiniCPM-V |  | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V) |
| MiniCPM-V-2 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm-v-2) | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2) |
| MiniCPM-Llama3-V-2_5 |  | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-Llama3-V-2_5) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |
| MiniCPM-V-2_6 | [link](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm-v-2_6) | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2_6) | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |
| StableDiffusion | | [link](../../python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion) |
| Bce-Embedding-Base-V1 | | | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/Embedding) |
| Speech_Paraformer-Large | | | [Python link](../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |
