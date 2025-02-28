# Intel® LLM Library for PyTorch*
<p>
  < <a href='./README.md'>English</a> | <b>中文 ></b> 
</p>

**`ipex-llm`** 是一个将大语言模型高效地运行于 Intel [GPU](docs/mddocs/Quickstart/install_windows_gpu.md) *(如搭载集成显卡的个人电脑，Arc 独立显卡、Flex 及 Max 数据中心 GPU 等)*、[NPU](docs/mddocs/Quickstart/npu_quickstart.md) 和 CPU 上的大模型 XPU 加速库[^1]。 
> [!NOTE]
> - *`ipex-llm`可以与  [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.zh-CN.md), [Ollama](docs/mddocs/Quickstart/ollama_portable_zip_quickstart.zh-CN.md), [HuggingFace transformers](python/llm/example/GPU/HuggingFace), [LangChain](python/llm/example/GPU/LangChain), [LlamaIndex](python/llm/example/GPU/LlamaIndex), [vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md), [Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md), [DeepSpeed-AutoTP](python/llm/example/GPU/Deepspeed-AutoTP), [FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md), [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md), [HuggingFace PEFT](python/llm/example/GPU/LLM-Finetuning), [HuggingFace TRL](python/llm/example/GPU/LLM-Finetuning/DPO), [AutoGen](python/llm/example/CPU/Applications/autogen), [ModeScope](python/llm/example/GPU/ModelScope-Models) 等无缝衔接。* 
> - ***70+** 模型已经在 `ipex-llm` 上得到优化和验证（如 Llama, Phi, Mistral, Mixtral, Whisper, DeepSeek, Qwen, ChatGLM, MiniCPM, Qwen-VL, MiniCPM-V 等）, 以获得先进的 **大模型算法优化**, **XPU 加速** 以及 **低比特（FP8FP8/FP6/FP4/INT4）支持**；更多模型信息请参阅[这里](#模型验证)。*

## 最新更新 🔥 
- [2025/02] 新增 [llama.cpp Portable Zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) 在 Intel [GPU](docs/mddocs/Quickstart/llamacpp_portable_zip_gpu_quickstart.md) 和 [NPU](docs/mddocs/Quickstart/llama_cpp_npu_portable_zip_quickstart.zh-CN.md) 上直接**免安装运行 llama.cpp**。
- [2025/02] 新增 [Ollama Portable Zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) 在 Intel GPU 上直接**免安装运行 Ollama** (包括 [Windows](docs/mddocs/Quickstart/ollama_portable_zip_quickstart.zh-CN.md#windows用户指南) 和 [Linux](docs/mddocs/Quickstart/ollama_portable_zip_quickstart.zh-CN.md#linux用户指南))。
- [2025/02] 新增在 Intel Arc GPUs 上运行 [vLLM 0.6.6](docs/mddocs/DockerGuides/vllm_docker_quickstart.md) 的支持。
- [2025/01] 新增在 Intel Arc [B580](docs/mddocs/Quickstart/bmg_quickstart.md) GPU 上运行 `ipex-llm` 的指南。
- [2025/01] 新增在 Intel GPU 上运行 [Ollama 0.5.4](docs/mddocs/Quickstart/ollama_quickstart.zh-CN.md) 的支持。
- [2024/12] 增加了对 Intel Core Ultra [NPU](docs/mddocs/Quickstart/npu_quickstart.md)（包括 100H，200V，200K 和 200H 系列）的 **Python** 和 **C++** 支持。

<details><summary>更多更新</summary>
<br/>

- [2024/11] 新增在 Intel Arc GPUs 上运行 [vLLM 0.6.2](docs/mddocs/DockerGuides/vllm_docker_quickstart.md) 的支持。
- [2024/07] 新增 Microsoft **GraphRAG** 的支持（使用运行在本地 Intel GPU 上的 LLM），详情参考[快速入门指南](docs/mddocs/Quickstart/graphrag_quickstart.md)。
- [2024/07] 全面增强了对多模态大模型的支持，包括 [StableDiffusion](python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion), [Phi-3-Vision](python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision), [Qwen-VL](python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl)，更多详情请点击[这里](python/llm/example/GPU/HuggingFace/Multimodal)。
- [2024/07] 新增 Intel GPU 上 **FP6** 的支持，详情参考[更多数据类型样例](python/llm/example/GPU/HuggingFace/More-Data-Types)。 
- [2024/06] 新增对 Intel Core Ultra 处理器中 **NPU** 的实验性支持，详情参考[相关示例](python/llm/example/NPU/HF-Transformers-AutoModels)。 
- [2024/06] 增加了对[流水线并行推理](python/llm/example/GPU/Pipeline-Parallel-Inference)的全面支持，使得用两块或更多 Intel GPU（如 Arc）上运行 LLM 变得更容易。
- [2024/06] 新增在 Intel GPU 上运行 **RAGFlow** 的支持，详情参考[快速入门指南](docs/mddocs/Quickstart/ragflow_quickstart.md)。
- [2024/05] 新增 **Axolotl** 的支持，可以在 Intel GPU 上进行LLM微调，详情参考[快速入门指南](docs/mddocs/Quickstart/axolotl_quickstart.md)。
- [2024/05] 你可以使用 **Docker** [images](#docker) 很容易地运行 `ipex-llm` 推理、服务和微调。
- [2024/05] 你能够在 Windows 上仅使用 "*[one command](docs/mddocs/Quickstart/install_windows_gpu.zh-CN.md#安装-ipex-llm)*" 来安装 `ipex-llm`。
- [2024/04] 你现在可以在 Intel GPU 上使用 `ipex-llm` 运行 **Open WebUI** ，详情参考[快速入门指南](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md)。
- [2024/04] 你现在可以在 Intel GPU 上使用 `ipex-llm` 以及 `llama.cpp` 和 `ollama` 运行 **Llama 3** ，详情参考[快速入门指南](docs/mddocs/Quickstart/llama3_llamacpp_ollama_quickstart.md)。
- [2024/04] `ipex-llm` 现在在Intel [GPU](python/llm/example/GPU/HuggingFace/LLM/llama3) 和 [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) 上都支持 **Llama 3** 了。
- [2024/04] `ipex-llm` 现在提供 C++ 推理, 在 Intel GPU 上它可以用作运行 [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.zh-CN.md) 和 [ollama](docs/mddocs/Quickstart/ollama_quickstart.zh-CN.md) 的加速后端。
- [2024/03] `bigdl-llm` 现已更名为 `ipex-llm` (请参阅[此处](docs/mddocs/Quickstart/bigdl_llm_migration.md)的迁移指南)，你可以在[这里](https://github.com/intel-analytics/bigdl-2.x)找到原始BigDL项目。
- [2024/02] `ipex-llm` 现在支持直接从 [ModelScope](python/llm/example/GPU/ModelScope-Models) ([魔搭](python/llm/example/CPU/ModelScope-Models)) loading 模型。
- [2024/02] `ipex-llm` 增加 **INT2** 的支持 (基于 llama.cpp [IQ2](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) 机制), 这使得在具有 16GB VRAM 的 Intel GPU 上运行大型 LLM（例如 Mixtral-8x7B）成为可能。
- [2024/02] 用户现在可以通过 [Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUI 使用 `ipex-llm`。
- [2024/02] `ipex-llm` 现在支持 *[Self-Speculative Decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md)*，这使得在 Intel [GPU](python/llm/example/GPU/Speculative-Decoding) 和 [CPU](python/llm/example/CPU/Speculative-Decoding) 上为 FP16 和 BF16 推理带来 **~30% 加速** 。
- [2024/02] `ipex-llm` 现在支持在 Intel GPU 上进行各种 LLM 微调(包括 [LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) 和 [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora))。
- [2024/01] 使用 `ipex-llm` [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA)，我们成功地在 8 个 Intel Max 1550 GPU 上使用 [Standford-Alpaca](python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora) 数据集分别对 LLaMA2-7B（**21 分钟内**）和 LLaMA2-70B（**3.14 小时内**）进行了微调，具体详情参阅[博客](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)。 
- [2023/12] `ipex-llm` 现在支持 [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora) (具体内容请参阅 *["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*).
- [2023/12] `ipex-llm` 现在在 Intel [GPU](python/llm/example/GPU/HuggingFace/LLM/mixtral) 和 [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral) 上均支持 [Mixtral-8x7B](python/llm/example/GPU/HuggingFace/LLM/mixtral)。
- [2023/12] `ipex-llm` 现在支持 [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) (具体内容请参阅 *["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*). 
- [2023/12] `ipex-llm` 现在在 Intel ***GPU*** 上支持 [FP8 and FP4 inference](python/llm/example/GPU/HuggingFace/More-Data-Types)。
- [2023/11] 初步支持直接将 [GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF)，[AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ) 和 [GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ) 模型加载到 `ipex-llm` 中。
- [2023/11] `ipex-llm` 现在在 Intel [GPU](python/llm/example/GPU/vLLM-Serving) 和 [CPU](python/llm/example/CPU/vLLM-Serving) 上都支持 [vLLM continuous batching](python/llm/example/GPU/vLLM-Serving) 。
- [2023/10] `ipex-llm` 现在在 Intel [GPU](python/llm/example/GPU/LLM-Finetuning/QLoRA) 和 [CPU](python/llm/example/CPU/QLoRA-FineTuning) 上均支持 [QLoRA finetuning](python/llm/example/GPU/LLM-Finetuning/QLoRA) 。
- [2023/10] `ipex-llm` 现在在 Intel GPU 和 CPU 上都支持 [FastChat serving](python/llm/src/ipex_llm/llm/serving) 。
- [2023/09] `ipex-llm` 现在支持 [Intel GPU](python/llm/example/GPU) (包括 iGPU, Arc, Flex 和 MAX)。
- [2023/09] `ipex-llm` [教程](https://github.com/intel-analytics/ipex-llm-tutorial) 已发布。
 
</details> 

## `ipex-llm` Demo

以下分别是使用 `ipex-llm` 在英特尔酷睿Ultra iGPU、酷睿Ultra NPU、单卡 Arc GPU 或双卡 Arc GPU 上运行本地 LLM 的 DEMO 演示，

<table width="100%">
  <tr>
    <td align="center" colspan="1"><strong>Intel Core Ultra iGPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Core Ultra NPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Arc dGPU</strong></td>
    <td align="center" colspan="1"><strong>2-Card Intel Arc dGPUs</strong></td>
  </tr>
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/npu_llama3.2-3B.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/npu_llama3.2-3B.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/2arc_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/2arc_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gif" width=100%; />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/ollama_quickstart.zh-CN.md">Ollama <br> (Mistral-7B, Q4_K) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/npu_quickstart.md">HuggingFace <br> (Llama3.2-3B, SYM_INT4)</a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/webui_quickstart.md">TextGeneration-WebUI <br> (Llama3-8B, FP8) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/llama_cpp_quickstart.zh-CN.md">llama.cpp <br> (DeepSeek-R1-Distill-Qwen-32B, Q4_K)</a>
    </td>  </tr>
</table>

<!--
See the demo of running [*Text-Generation-WebUI*](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html), [*local RAG using LangChain-Chatchat*](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/chatchat_quickstart.html), [*llama.cpp*](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html) and [*Ollama*](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html) *(on either Intel Core Ultra laptop or Arc GPU)* with `ipex-llm`  below.

<table width="100%">
  <tr>
    <td align="center" colspan="2"><strong>Intel Core Ultra Laptop</strong></td>
    <td align="center" colspan="2"><strong>Intel Arc GPU</strong></td>
  </tr>
  <tr>
    <td>
      <video src="https://private-user-images.githubusercontent.com/1931082/319632616-895d56cd-e74b-4da1-b4d1-2157df341424.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIyNDE4MjUsIm5iZiI6MTcxMjI0MTUyNSwicGF0aCI6Ii8xOTMxMDgyLzMxOTYzMjYxNi04OTVkNTZjZC1lNzRiLTRkYTEtYjRkMS0yMTU3ZGYzNDE0MjQubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQwNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MDRUMTQzODQ1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9Y2JmYzkxYWFhMGYyN2MxYTkxOTI3MGQ2NTFkZDY4ZjFjYjg3NmZhY2VkMzVhZTU2OGEyYjhjNzI5YTFhOGNhNSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.Ga8mmCAO62DFCNzU1fdoyC_4MzqhDHzjZedzmi_2L-I" width=100% controls />
    </td>
    <td>
      <video src="https://private-user-images.githubusercontent.com/1931082/319625142-68da379e-59c6-4308-88e8-c17e40baba7b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIyNDA2MzQsIm5iZiI6MTcxMjI0MDMzNCwicGF0aCI6Ii8xOTMxMDgyLzMxOTYyNTE0Mi02OGRhMzc5ZS01OWM2LTQzMDgtODhlOC1jMTdlNDBiYWJhN2IubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQwNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MDRUMTQxODU0WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NzYwOWI4MmQxZjFhMjJlNGNhZTA3MGUyZDE4OTA0N2Q2YjQ4NTcwN2M2MTY1ODAwZmE3OTIzOWI0Y2U3YzYwNyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.g0bYAj3J8IJci7pLzoJI6QDalyzXzMYtQkDY7aqZMc4" width=100% controls />
    </td>
    <td>
      <video src="https://private-user-images.githubusercontent.com/1931082/319625685-ff13b099-bcda-48f1-b11b-05421e7d386d.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIyNDA4MTcsIm5iZiI6MTcxMjI0MDUxNywicGF0aCI6Ii8xOTMxMDgyLzMxOTYyNTY4NS1mZjEzYjA5OS1iY2RhLTQ4ZjEtYjExYi0wNTQyMWU3ZDM4NmQubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQwNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MDRUMTQyMTU3WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MWQ3MmEwZGRkNGVlY2RkNjAzMTliODM1NDEzODU3NWQ0ZGE4MjYyOGEyZjdkMjBiZjI0MjllYTU4ODQ4YzM0NCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.OFxex8Yj6WyqJKMi6B1Q19KkmbYqYCg1rD49wUwxdXQ" width=100% controls />
    </td>
    <td>
      <video src="https://private-user-images.githubusercontent.com/1931082/325939544-2fc0ad5e-9ac7-4f95-b7b9-7885a8738443.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQxMjYwODAsIm5iZiI6MTcxNDEyNTc4MCwicGF0aCI6Ii8xOTMxMDgyLzMyNTkzOTU0NC0yZmMwYWQ1ZS05YWM3LTRmOTUtYjdiOS03ODg1YTg3Mzg0NDMubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQyNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MjZUMTAwMzAwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YjZlZDE4YjFjZWJkMzQ4NmY3ZjNlMmRiYWUzMDYxMTI3YzcxYjRiYjgwNmE2NDliMjMwOTI0NWJhMDQ1NDY1YyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.WfA2qwr8EP9W7a3oOYcKqaqsEKDlAkF254zbmn9dVv0" width=100% controls />
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html">Text-Generation-WebUI</a>
    </td>
    <td align="center" width="25%">
      <a href="https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/chatchat_quickstart.html">Local RAG using LangChain-Chatchat</a>
    </td>
    <td align="center" width="25%">
      <a href="https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html">llama.cpp</a>
    </td>
    <td align="center" width="25%">
      <a href="https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html">Ollama</a>
    </td>  </tr>
</table>
-->

## `ipex-llm` 性能
下图展示了在 Intel Core Ultra 和 Intel Arc GPU 上的 **Token 生成速度**[^1]（更多详情可点击 [[2]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-meta-llama3-with-intel-ai-solutions.html)[[3]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-microsoft-phi-3-models-intel-ai-soln.html)[[4]](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-ai-solutions-accelerate-alibaba-qwen2-llms.html))。

<table width="100%">
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/MTL_perf.jpg" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/MTL_perf.jpg" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/Arc_perf.jpg" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/Arc_perf.jpg" width=100%; />
      </a>
    </td>
  </tr>
</table>

如果需要自己进行 `ipex-llm` 性能基准测试，可参考[基准测试指南](docs/mddocs/Quickstart/benchmark_quickstart.md)。

## 模型准确率
部分模型的 **Perplexity** 结果如下所示（使用 Wikitext 数据集和[此处](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/dev/benchmark/perplexity)的脚本进行测试)。
|Perplexity                 |sym_int4	|q4_k	  |fp6	  |fp8_e5m2 |fp8_e4m3 |fp16   |
|---------------------------|---------|-------|-------|---------|---------|-------|
|Llama-2-7B-chat-hf	        |6.364 	  |6.218 	|6.092 	|6.180 	  |6.098    |6.096  | 
|Mistral-7B-Instruct-v0.2	  |5.365 	  |5.320 	|5.270 	|5.273 	  |5.246	   |5.244  |
|Baichuan2-7B-chat	         |6.734    |6.727	 |6.527	 |6.539	   |6.488	   |6.508  |
|Qwen1.5-7B-chat	           |8.865 	  |8.816 	|8.557 	|8.846 	  |8.530    |8.607  | 
|Llama-3.1-8B-Instruct	     |6.705	   |6.566	 |6.338	 |6.383	   |6.325	   |6.267  |
|gemma-2-9b-it	             |7.541	   |7.412	 |7.269	 |7.380	   |7.268	   |7.270  |
|Baichuan2-13B-Chat	        |6.313	   |6.160	 |6.070	 |6.145	   |6.086	   |6.031  |
|Llama-2-13b-chat-hf	       |5.449	   |5.422	 |5.341	 |5.384	   |5.332	   |5.329  |
|Qwen1.5-14B-Chat	          |7.529	   |7.520	 |7.367	 |7.504	   |7.297	   |7.334  |

[^1]: Performance varies by use, configuration and other factors. `ipex-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex

## `ipex-llm` 快速入门

### 使用
- [Ollama Portable Zip](docs/mddocs/Quickstart/ollama_portable_zip_quickstart.zh-CN.md): 在 Intel GPU 上直接**免安装运行 Ollama**。
- [Arc B580](docs/mddocs/Quickstart/bmg_quickstart.md): 在 Intel Arc **B580** GPU 上运行 `ipex-llm`（包括 Ollama, llama.cpp, PyTorch, HuggingFace 等）
- [NPU](docs/mddocs/Quickstart/npu_quickstart.md): 在 Intel **NPU** 上运行 `ipex-llm`（支持 Python 和 C++）
- [Ollama](docs/mddocs/Quickstart/ollama_quickstart.zh-CN.md): 在 Intel GPU 上运行 **ollama** (*使用 `ipex-llm` 的 C++ 接口*) 
- [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.zh-CN.md): 在 Intel GPU 上运行 **llama.cpp** (*使用 `ipex-llm` 的 C++ 接口*) 
- [PyTorch/HuggingFace](docs/mddocs/Quickstart/install_windows_gpu.zh-CN.md): 使用 [Windows](docs/mddocs/Quickstart/install_windows_gpu.zh-CN.md) 和 [Linux](docs/mddocs/Quickstart/install_linux_gpu.zh-CN.md) 在 Intel GPU 上运行 **PyTorch**、**HuggingFace**、**LangChain**、**LlamaIndex** 等 (*使用 `ipex-llm` 的 Python 接口*) 
- [vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md): 在 Intel [GPU](docs/mddocs/DockerGuides/vllm_docker_quickstart.md) 和 [CPU](docs/mddocs/DockerGuides/vllm_cpu_docker_quickstart.md) 上使用 `ipex-llm` 运行 **vLLM** 
- [FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md): 在 Intel GPU 和 CPU 上使用 `ipex-llm` 运行 **FastChat** 服务
- [Serving on multiple Intel GPUs](docs/mddocs/Quickstart/deepspeed_autotp_fastapi_quickstart.md): 利用 DeepSpeed AutoTP 和 FastAPI 在 **多个 Intel GPU** 上运行 `ipex-llm` 推理服务
- [Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md): 使用 `ipex-llm` 运行 `oobabooga` **WebUI** 
- [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md): 使用 **Axolotl** 和 `ipex-llm` 进行 LLM 微调
- [Benchmarking](docs/mddocs/Quickstart/benchmark_quickstart.md):  在 Intel GPU 和 CPU 上运行**性能基准测试**（延迟和吞吐量）

### Docker
- [GPU Inference in C++](docs/mddocs/DockerGuides/docker_cpp_xpu_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `llama.cpp`, `ollama`等
- [GPU Inference in Python](docs/mddocs/DockerGuides/docker_pytorch_inference_gpu.md) : 在 Intel GPU 上使用 `ipex-llm` 运行 HuggingFace `transformers`, `LangChain`, `LlamaIndex`, `ModelScope`，等
- [vLLM on GPU](docs/mddocs/DockerGuides/vllm_docker_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `vLLM` 推理服务
- [vLLM on CPU](docs/mddocs/DockerGuides/vllm_cpu_docker_quickstart.md): 在 Intel CPU 上使用 `ipex-llm` 运行 `vLLM` 推理服务  
- [FastChat on GPU](docs/mddocs/DockerGuides/fastchat_docker_quickstart.md): 在 Intel GPU 上使用 `ipex-llm` 运行 `FastChat` 推理服务
- [VSCode on GPU](docs/mddocs/DockerGuides/docker_run_pytorch_inference_in_vscode.md): 在 Intel GPU 上使用 VSCode 开发并运行基于 Python 的 `ipex-llm` 应用

### 应用
- [GraphRAG](docs/mddocs/Quickstart/graphrag_quickstart.md): 基于 `ipex-llm` 使用本地 LLM 运行 Microsoft 的 `GraphRAG`
- [RAGFlow](docs/mddocs/Quickstart/ragflow_quickstart.md): 基于 `ipex-llm` 运行 `RAGFlow` (*一个开源的 RAG 引擎*)  
- [LangChain-Chatchat](docs/mddocs/Quickstart/chatchat_quickstart.md): 基于 `ipex-llm` 运行 `LangChain-Chatchat` (*使用 RAG pipline 的知识问答库*)
- [Coding copilot](docs/mddocs/Quickstart/continue_quickstart.md): 基于 `ipex-llm` 运行 `Continue` (VSCode 里的编码智能助手)
- [Open WebUI](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md): 基于 `ipex-llm` 运行 `Open WebUI`
- [PrivateGPT](docs/mddocs/Quickstart/privateGPT_quickstart.md): 基于 `ipex-llm` 运行 `PrivateGPT` 与文档进行交互
- [Dify platform](docs/mddocs/Quickstart/dify_quickstart.md): 在`Dify`(*一款开源的大语言模型应用开发平台*) 里接入 `ipex-llm` 加速本地 LLM

### 安装
- [Windows GPU](docs/mddocs/Quickstart/install_windows_gpu.zh-CN.md): 在带有 Intel GPU 的 Windows 系统上安装 `ipex-llm` 
- [Linux GPU](docs/mddocs/Quickstart/install_linux_gpu.zh-CN.md): 在带有 Intel GPU 的Linux系统上安装 `ipex-llm` 
- *更多内容, 请参考[完整安装指南](docs/mddocs/Overview/install.md)*

### 代码示例
- #### 低比特推理
  - [INT4 inference](python/llm/example/GPU/HuggingFace/LLM): 在 Intel [GPU](python/llm/example/GPU/HuggingFace/LLM) 和 [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model) 上进行 **INT4** LLM 推理
  - [FP8/FP6/FP4 inference](python/llm/example/GPU/HuggingFace/More-Data-Types): 在 Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types) 上进行 **FP8**，**FP6** 和 **FP4** LLM 推理
  - [INT8 inference](python/llm/example/GPU/HuggingFace/More-Data-Types): 在 Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types) 和 [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types) 上进行 **INT8** LLM 推理 
  - [INT2 inference](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2): 在 Intel [GPU](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) 上进行 **INT2** LLM 推理 (基于 llama.cpp IQ2 机制) 
- #### FP16/BF16 推理
  - 在 Intel [GPU](python/llm/example/GPU/Speculative-Decoding) 上进行 **FP16** LLM 推理（并使用 [self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md) 优化）
  - 在 Intel [CPU](python/llm/example/CPU/Speculative-Decoding) 上进行 **BF16** LLM 推理（并使用 [self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md) 优化）
- #### 分布式推理
  - 在 Intel [GPU](python/llm/example/GPU/Pipeline-Parallel-Inference) 上进行 **流水线并行** 推理
  - 在 Intel [GPU](python/llm/example/GPU/Deepspeed-AutoTP) 上进行 **DeepSpeed AutoTP** 推理
- #### 保存和加载
  - [Low-bit models](python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load): 保存和加载 `ipex-llm` 低比特模型 (INT4/FP4/FP6/INT8/FP8/FP16/etc.)
  - [GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF): 直接将 GGUF 模型加载到 `ipex-llm` 中
  - [AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ): 直接将 AWQ 模型加载到 `ipex-llm` 中
  - [GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ): 直接将 GPTQ 模型加载到 `ipex-llm` 中
- #### 微调
  - 在 Intel [GPU](python/llm/example/GPU/LLM-Finetuning) 进行 LLM 微调，包括 [LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA)，[QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA)，[DPO](python/llm/example/GPU/LLM-Finetuning/DPO)，[QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) 和 [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)
  - 在 Intel [CPU](python/llm/example/CPU/QLoRA-FineTuning) 进行 QLoRA 微调 
- #### 与社区库集成
  - [HuggingFace transformers](python/llm/example/GPU/HuggingFace)
  - [Standard PyTorch model](python/llm/example/GPU/PyTorch-Models)
  - [LangChain](python/llm/example/GPU/LangChain)
  - [LlamaIndex](python/llm/example/GPU/LlamaIndex)
  - [DeepSpeed-AutoTP](python/llm/example/GPU/Deepspeed-AutoTP)
  - [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md)
  - [HuggingFace PEFT](python/llm/example/GPU/LLM-Finetuning/HF-PEFT)
  - [HuggingFace TRL](python/llm/example/GPU/LLM-Finetuning/DPO)
  - [AutoGen](python/llm/example/CPU/Applications/autogen)
  - [ModeScope](python/llm/example/GPU/ModelScope-Models)
- [教程](https://github.com/intel-analytics/ipex-llm-tutorial)

## API 文档
- [HuggingFace Transformers 兼容的 API (Auto Classes)](docs/mddocs/PythonAPI/transformers.md)
- [适用于任意 Pytorch 模型的 API](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/PythonAPI/optimize.md)

## FAQ
- [常见问题解答](docs/mddocs/Overview/FAQ/faq.md)

## 模型验证
50+ 模型已经在 `ipex-llm` 上得到优化和验证，包括 *LLaMA/LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM2/ChatGLM3, Baichuan/Baichuan2, Qwen/Qwen-1.5, InternLM,* 更多模型请参看下表，
  
| 模型       | CPU 示例                                  | GPU 示例                                  | NPU 示例                                  |
|----------- |------------------------------------------|-------------------------------------------|-------------------------------------------|
| LLaMA  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](python/llm/example/GPU/HuggingFace/LLM/vicuna)|
| LLaMA 2    | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link](python/llm/example/GPU/HuggingFace/LLM/llama2)  | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3)  | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3.1    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3.1) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3.1)  |
| LLaMA 3.2    |  | [link](python/llm/example/GPU/HuggingFace/LLM/llama3.2)  | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| LLaMA 3.2-Vision    |  | [link](python/llm/example/GPU/PyTorch-Models/Model/llama3.2-vision/)  |
| ChatGLM    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |    | 
| ChatGLM2   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm2)   |
| ChatGLM3   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm3)   |
| GLM-4      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm4)      | [link](python/llm/example/GPU/HuggingFace/LLM/glm4)       |
| GLM-4V     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm-4v)    | [link](python/llm/example/GPU/HuggingFace/Multimodal/glm-4v)     |
| GLM-Edge   |  | [link](python/llm/example/GPU/HuggingFace/LLM/glm-edge) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM) |
| GLM-Edge-V   |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/glm-edge-v) |
| Mistral    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral)   | [link](python/llm/example/GPU/HuggingFace/LLM/mistral)    |
| Mixtral    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral)   | [link](python/llm/example/GPU/HuggingFace/LLM/mixtral)    |
| Falcon     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon)    | [link](python/llm/example/GPU/HuggingFace/LLM/falcon)     |
| MPT        | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt)       | [link](python/llm/example/GPU/HuggingFace/LLM/mpt)        |
| Dolly-v1   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)  | [link](python/llm/example/GPU/HuggingFace/LLM/dolly-v1)   | 
| Dolly-v2   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)  | [link](python/llm/example/GPU/HuggingFace/LLM/dolly-v2)   | 
| Replit Code| [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit)    | [link](python/llm/example/GPU/HuggingFace/LLM/replit)     |
| RedPajama  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama) |    | 
| Phoenix    | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix)   |    | 
| StarCoder  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder) | [link](python/llm/example/GPU/HuggingFace/LLM/starcoder) | 
| Baichuan   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan)  | [link](python/llm/example/GPU/HuggingFace/LLM/baichuan)   |
| Baichuan2  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](python/llm/example/GPU/HuggingFace/LLM/baichuan2)  | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM) |
| InternLM   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm)  | [link](python/llm/example/GPU/HuggingFace/LLM/internlm)   |
| InternVL2   |   | [link](python/llm/example/GPU/HuggingFace/Multimodal/internvl2)   |
| Qwen       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen)      | [link](python/llm/example/GPU/HuggingFace/LLM/qwen)       |
| Qwen1.5 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen1.5) |
| Qwen2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen2) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen2) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| Qwen2.5 |  | [link](python/llm/example/GPU/HuggingFace/LLM/qwen2.5) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| Qwen-VL    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen-vl)   | [link](python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl)    |
| Qwen2-VL    || [link](python/llm/example/GPU/HuggingFace/Multimodal/qwen2-vl)    |
| Qwen2-Audio    |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/qwen2-audio)    |
| Aquila     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila)    | [link](python/llm/example/GPU/HuggingFace/LLM/aquila)     |
| Aquila2     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila2)    | [link](python/llm/example/GPU/HuggingFace/LLM/aquila2)     |
| MOSS       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss)      |    | 
| Whisper    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper)   | [link](python/llm/example/GPU/HuggingFace/Multimodal/whisper)    |
| Phi-1_5    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-1_5)   | [link](python/llm/example/GPU/HuggingFace/LLM/phi-1_5)    |
| Flan-t5    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/flan-t5)   | [link](python/llm/example/GPU/HuggingFace/LLM/flan-t5)    |
| LLaVA      | [link](python/llm/example/CPU/PyTorch-Models/Model/llava)                 | [link](python/llm/example/GPU/PyTorch-Models/Model/llava)                  |
| CodeLlama  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codellama) | [link](python/llm/example/GPU/HuggingFace/LLM/codellama)  |
| Skywork      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/skywork)                 |    |
| InternLM-XComposer  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm-xcomposer)   |    |
| WizardCoder-Python | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/wizardcoder-python) | |
| CodeShell | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codeshell) | |
| Fuyu      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/fuyu) | |
| Distil-Whisper | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/distil-whisper) | [link](python/llm/example/GPU/HuggingFace/Multimodal/distil-whisper) |
| Yi | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/yi) | [link](python/llm/example/GPU/HuggingFace/LLM/yi) |
| BlueLM | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/bluelm) | [link](python/llm/example/GPU/HuggingFace/LLM/bluelm) |
| Mamba | [link](python/llm/example/CPU/PyTorch-Models/Model/mamba) | [link](python/llm/example/GPU/PyTorch-Models/Model/mamba) |
| SOLAR | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/solar) | [link](python/llm/example/GPU/HuggingFace/LLM/solar) |
| Phixtral | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phixtral) | [link](python/llm/example/GPU/HuggingFace/LLM/phixtral) |
| InternLM2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm2) | [link](python/llm/example/GPU/HuggingFace/LLM/internlm2) |
| RWKV4 |  | [link](python/llm/example/GPU/HuggingFace/LLM/rwkv4) |
| RWKV5 |  | [link](python/llm/example/GPU/HuggingFace/LLM/rwkv5) |
| Bark | [link](python/llm/example/CPU/PyTorch-Models/Model/bark) | [link](python/llm/example/GPU/PyTorch-Models/Model/bark) |
| SpeechT5 |  | [link](python/llm/example/GPU/PyTorch-Models/Model/speech-t5) |
| DeepSeek-MoE | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek-moe) |  |
| Ziya-Coding-34B-v1.0 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/ziya) | |
| Phi-2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-2) | [link](python/llm/example/GPU/HuggingFace/LLM/phi-2) |
| Phi-3 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-3) | [link](python/llm/example/GPU/HuggingFace/LLM/phi-3) |
| Phi-3-vision | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-3-vision) | [link](python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision) |
| Yuan2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/yuan2) | [link](python/llm/example/GPU/HuggingFace/LLM/yuan2) |
| Gemma | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/gemma) | [link](python/llm/example/GPU/HuggingFace/LLM/gemma) |
| Gemma2 |  | [link](python/llm/example/GPU/HuggingFace/LLM/gemma2) |
| DeciLM-7B | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deciLM-7b) | [link](python/llm/example/GPU/HuggingFace/LLM/deciLM-7b) |
| Deepseek | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek) | [link](python/llm/example/GPU/HuggingFace/LLM/deepseek) |
| StableLM | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/stablelm) | [link](python/llm/example/GPU/HuggingFace/LLM/stablelm) |
| CodeGemma | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codegemma) | [link](python/llm/example/GPU/HuggingFace/LLM/codegemma) |
| Command-R/cohere | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/cohere) | [link](python/llm/example/GPU/HuggingFace/LLM/cohere) |
| CodeGeeX2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codegeex2) | [link](python/llm/example/GPU/HuggingFace/LLM/codegeex2) |
| MiniCPM | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm) | [link](python/llm/example/GPU/HuggingFace/LLM/minicpm) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM), [C++ link](python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) |
| MiniCPM3 |  | [link](python/llm/example/GPU/HuggingFace/LLM/minicpm3) |
| MiniCPM-V |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V) |
| MiniCPM-V-2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm-v-2) | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2) |
| MiniCPM-Llama3-V-2_5 |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-Llama3-V-2_5) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |
| MiniCPM-V-2_6 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm-v-2_6) | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2_6) | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |
| MiniCPM-o-2_6 | | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-o-2_6/) |
| Janus-Pro | | [link](python/llm/example/GPU/HuggingFace/Multimodal/janus-pro/) |
| StableDiffusion | | [link](python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion) |
| Bce-Embedding-Base-V1 | | | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/Embedding) |
| Speech_Paraformer-Large | | | [Python link](python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal) |

## 官方支持
- 如果遇到问题，或者请求新功能支持，请提交 [Github Issue](https://github.com/intel-analytics/ipex-llm/issues) 告诉我们
- 如果发现漏洞，请在 [GitHub Security Advisory](https://github.com/intel-analytics/ipex-llm/security/advisories) 提交漏洞报告
