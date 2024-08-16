> [!IMPORTANT]
> ***`bigdl-llm` has now become `ipex-llm` (see the migration guide [here](docs/mddocs/Quickstart/bigdl_llm_migration.md)); you may find the original `BigDL` project [here](https://github.com/intel-analytics/BigDL-2.x).***
 
---

# 💫 Intel® LLM library for PyTorch*
**`IPEX-LLM`** is a PyTorch library for running **LLM** on Intel CPU and GPU *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)* with very low latency[^1]. 
> [!NOTE]
> - *It is built on top of the excellent work of **`llama.cpp`**, **`transformers`**, **`bitsandbytes`**, **`vLLM`**, **`qlora`**, **`AutoGPTQ`**, **`AutoAWQ`**, etc.*
> - *It provides seamless integration with [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md), [Ollama](docs/mddocs/Quickstart/ollama_quickstart.md), [Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md), [HuggingFace transformers](python/llm/example/GPU/HuggingFace), [LangChain](python/llm/example/GPU/LangChain), [LlamaIndex](python/llm/example/GPU/LlamaIndex), [DeepSpeed-AutoTP](python/llm/example/GPU/Deepspeed-AutoTP), [vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md), [FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md), [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md), [HuggingFace PEFT](python/llm/example/GPU/LLM-Finetuning), [HuggingFace TRL](python/llm/example/GPU/LLM-Finetuning/DPO), [AutoGen](python/llm/example/CPU/Applications/autogen), [ModeScope](python/llm/example/GPU/ModelScope-Models), etc.* 
> - ***50+ models** have been optimized/verified on `ipex-llm` (including LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM, Baichuan, Qwen, RWKV, and more); see the complete list [here](#verified-models).*

## Latest Update 🔥 
- [2024/07] We added support for running Microsoft's **GraphRAG** using local LLM on Intel GPU; see the quickstart guide [here](docs/mddocs/Quickstart/graphrag_quickstart.md).
- [2024/07] We added extensive support for Large Multimodal Models, including [StableDiffusion](https://github.com/jason-dai/ipex-llm/tree/main/python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion), [Phi-3-Vision](python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision), [Qwen-VL](python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl), and [more](python/llm/example/GPU/HuggingFace/Multimodal).
- [2024/07] We added **FP6** support on Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types). 
- [2024/06] We added experimental **NPU** support for Intel Core Ultra processors; see the examples [here](python/llm/example/NPU/HF-Transformers-AutoModels). 
- [2024/06] We added extensive support of **pipeline parallel** [inference](python/llm/example/GPU/Pipeline-Parallel-Inference), which makes it easy to run large-sized LLM using 2 or more Intel GPUs (such as Arc).
- [2024/06] We added support for running **RAGFlow** with `ipex-llm` on Intel [GPU](docs/mddocs/Quickstart/ragflow_quickstart.md).
- [2024/05] `ipex-llm` now supports **Axolotl** for LLM finetuning on Intel GPU; see the quickstart [here](docs/mddocs/Quickstart/axolotl_quickstart.md).

<details><summary>More updates</summary>
<br/>
 
- [2024/05] You can now easily run `ipex-llm` inference, serving and finetuning using the **Docker** [images](#docker).
- [2024/05] You can now install `ipex-llm` on Windows using just "*[one command](docs/mddocs/Quickstart/install_windows_gpu.md#install-ipex-llm)*".
- [2024/04] You can now run **Open WebUI** on Intel GPU using `ipex-llm`; see the quickstart [here](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md).
- [2024/04] You can now run **Llama 3** on Intel GPU using `llama.cpp` and `ollama` with `ipex-llm`; see the quickstart [here](docs/mddocs/Quickstart/llama3_llamacpp_ollama_quickstart.md).
- [2024/04] `ipex-llm` now supports **Llama 3** on both Intel [GPU](python/llm/example/GPU/HuggingFace/LLM/llama3) and [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3).
- [2024/04] `ipex-llm` now provides C++ interface, which can be used as an accelerated backend for running [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md) and [ollama](docs/mddocs/Quickstart/ollama_quickstart.md) on Intel GPU.
- [2024/03] `bigdl-llm` has now become `ipex-llm` (see the migration guide [here](docs/mddocs/Quickstart/bigdl_llm_migration.md)); you may find the original `BigDL` project [here](https://github.com/intel-analytics/bigdl-2.x).
- [2024/02] `ipex-llm` now supports directly loading model from [ModelScope](python/llm/example/GPU/ModelScope-Models) ([魔搭](python/llm/example/CPU/ModelScope-Models)).
- [2024/02] `ipex-llm` added initial **INT2** support (based on llama.cpp [IQ2](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) mechanism), which makes it possible to run large-sized LLM (e.g., Mixtral-8x7B) on Intel GPU with 16GB VRAM.
- [2024/02] Users can now use `ipex-llm` through [Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUI.
- [2024/02] `ipex-llm` now supports *[Self-Speculative Decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md)*, which in practice brings **~30% speedup** for FP16 and BF16 inference latency on Intel [GPU](python/llm/example/GPU/Speculative-Decoding) and [CPU](python/llm/example/CPU/Speculative-Decoding) respectively.
- [2024/02] `ipex-llm` now supports a comprehensive list of LLM **finetuning** on Intel GPU (including [LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) and [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)).
- [2024/01] Using `ipex-llm` [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), we managed to finetune LLaMA2-7B in **21 minutes** and LLaMA2-70B in **3.14 hours** on 8 Intel Max 1550 GPU for [Standford-Alpaca](python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora) (see the blog [here](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)). 
- [2023/12] `ipex-llm` now supports [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora) (see *["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*).
- [2023/12] `ipex-llm` now supports [Mixtral-8x7B](python/llm/example/GPU/HuggingFace/LLM/mixtral) on both Intel [GPU](python/llm/example/GPU/HuggingFace/LLM/mixtral) and [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral). 
- [2023/12] `ipex-llm` now supports [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) (see *["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*). 
- [2023/12] `ipex-llm` now supports [FP8 and FP4 inference](python/llm/example/GPU/HuggingFace/More-Data-Types) on Intel ***GPU***.
- [2023/11] Initial support for directly loading [GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF), [AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ) and [GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ) models into `ipex-llm` is available.
- [2023/11] `ipex-llm` now supports [vLLM continuous batching](python/llm/example/GPU/vLLM-Serving) on both Intel [GPU](python/llm/example/GPU/vLLM-Serving) and [CPU](python/llm/example/CPU/vLLM-Serving).
- [2023/10] `ipex-llm` now supports [QLoRA finetuning](python/llm/example/GPU/LLM-Finetuning/QLoRA) on both Intel [GPU](python/llm/example/GPU/LLM-Finetuning/QLoRA) and [CPU](python/llm/example/CPU/QLoRA-FineTuning).
- [2023/10] `ipex-llm` now supports [FastChat serving](python/llm/src/ipex_llm/llm/serving) on on both Intel CPU and GPU.
- [2023/09] `ipex-llm` now supports [Intel GPU](python/llm/example/GPU) (including iGPU, Arc, Flex and MAX).
- [2023/09] `ipex-llm` [tutorial](https://github.com/intel-analytics/ipex-llm-tutorial) is released.
 
</details> 

## `ipex-llm` Performance
See the **Token Generation Speed** on *Intel Core Ultra* and *Intel Arc GPU* below[^1] (and refer to [[2]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-meta-llama3-with-intel-ai-solutions.html)[[3]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-microsoft-phi-3-models-intel-ai-soln.html)[[4]](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-ai-solutions-accelerate-alibaba-qwen2-llms.html) for more details).

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

You may follow the [Benchmarking Guide](docs/mddocs/Quickstart/benchmark_quickstart.md) to run `ipex-llm` performance benchmark yourself.

## `ipex-llm` Demo

See demos of running local LLMs *on Intel Iris iGPU, Intel Core Ultra iGPU, single-card Arc GPU, or multi-card Arc GPUs* using `ipex-llm` below.

<table width="100%">
  <tr>
    <td align="center" colspan="1"><strong>Intel Iris iGPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Core Ultra iGPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Arc dGPU</strong></td>
    <td align="center" colspan="1"><strong>2-Card Intel Arc dGPUs</strong></td>
  </tr>
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/iris_phi3-3.8B_q4_0_llamacpp_long.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/iris_phi3-3.8B_q4_0_llamacpp_long.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/2arc_qwen1.5-32B_fp6_fastchat.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/2arc_qwen1.5-32B_fp6_fastchat.gif" width=100%; />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/llama_cpp_quickstart.md">llama.cpp (Phi-3-mini Q4_0)</a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/ollama_quickstart.md">Ollama (Mistral-7B Q4_K) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/webui_quickstart.md">TextGeneration-WebUI (Llama3-8B FP8) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/fastchat_quickstart.md">FastChat (QWen1.5-32B FP6)</a>
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

## Model Accuracy
Please see the **Perplexity** result below (tested on Wikitext dataset using the script [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/dev/benchmark/perplexity)).
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

[^1]: Performance varies by use, configuration and other factors. `ipex-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.

## `ipex-llm` Quickstart

### Docker
- [GPU Inference in C++](docs/mddocs/DockerGuides/docker_cpp_xpu_quickstart.md): running `llama.cpp`, `ollama`, `OpenWebUI`, etc., with `ipex-llm` on Intel GPU
- [GPU Inference in Python](docs/mddocs/DockerGuides/docker_pytorch_inference_gpu.md) : running HuggingFace `transformers`, `LangChain`, `LlamaIndex`, `ModelScope`, etc. with `ipex-llm` on Intel GPU
- [vLLM on GPU](docs/mddocs/DockerGuides/vllm_docker_quickstart.md): running `vLLM` serving with `ipex-llm` on Intel GPU
- [vLLM on CPU](docs/mddocs/DockerGuides/vllm_cpu_docker_quickstart.md): running `vLLM` serving with `ipex-llm` on Intel CPU  
- [FastChat on GPU](docs/mddocs/DockerGuides/fastchat_docker_quickstart.md): running `FastChat` serving with `ipex-llm` on Intel GPU
- [VSCode on GPU](docs/mddocs/DockerGuides/docker_run_pytorch_inference_in_vscode.md): running and developing `ipex-llm` applications in Python using VSCode on Intel GPU

### Use
- [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md): running **llama.cpp** (*using C++ interface of `ipex-llm` as an accelerated backend for `llama.cpp`*) on Intel GPU
- [Ollama](docs/mddocs/Quickstart/ollama_quickstart.md): running **ollama** (*using C++ interface of `ipex-llm` as an accelerated backend for `ollama`*) on Intel GPU
- [Llama 3 with `llama.cpp` and `ollama`](docs/mddocs/Quickstart/llama3_llamacpp_ollama_quickstart.md): running **Llama 3** on Intel GPU using `llama.cpp` and `ollama` with `ipex-llm`
- [vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md): running `ipex-llm` in **vLLM** on both Intel [GPU](python/llm/example/GPU/vLLM-Serving) and [CPU](python/llm/example/CPU/vLLM-Serving)
- [FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md): running `ipex-llm` in **FastChat** serving on on both Intel GPU and CPU
- [Serving on multiple Intel GPUs](docs/mddocs/Quickstart/deepspeed_autotp_fastapi_quickstart.md): running `ipex-llm` **serving on multiple Intel GPUs** by leveraging DeepSpeed AutoTP and FastAPI
- [Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md): running `ipex-llm` in `oobabooga` **WebUI**
- [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md): running `ipex-llm` in **Axolotl** for LLM finetuning
- [Benchmarking](docs/mddocs/Quickstart/benchmark_quickstart.md): running  (latency and throughput) **benchmarks** for `ipex-llm` on Intel CPU and GPU

### Applications
- [GraphRAG](docs/mddocs/Quickstart/graphrag_quickstart.md): running Microsoft's `GraphRAG` using local LLM with `ipex-llm`
- [RAGFlow](docs/mddocs/Quickstart/ragflow_quickstart.md): running `RAGFlow` (*an open-source RAG engine*) with `ipex-llm`
- [LangChain-Chatchat](docs/mddocs/Quickstart/chatchat_quickstart.md): running `LangChain-Chatchat` (*Knowledge Base QA using RAG pipeline*) with `ipex-llm`
- [Coding copilot](docs/mddocs/Quickstart/continue_quickstart.md): running `Continue` (coding copilot in VSCode) with `ipex-llm`
- [Open WebUI](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md): running `Open WebUI` with `ipex-llm`
- [PrivateGPT](docs/mddocs/Quickstart/privateGPT_quickstart.md): running `PrivateGPT` to interact with documents with `ipex-llm`
- [Dify platform](docs/mddocs/Quickstart/dify_quickstart.md): running `ipex-llm` in `Dify`(*production-ready LLM app development platform*)

### Install 
- [Windows GPU](docs/mddocs/Quickstart/install_windows_gpu.md): installing `ipex-llm` on Windows with Intel GPU
- [Linux GPU](docs/mddocs/Quickstart/install_linux_gpu.md): installing `ipex-llm` on Linux with Intel GPU
- *For more details, please refer to the [full installation guide](docs/mddocs/Overview/install.md)*

### Code Examples
- #### Low bit inference
  - [INT4 inference](python/llm/example/GPU/HuggingFace/LLM): **INT4** LLM inference on Intel [GPU](python/llm/example/GPU/HuggingFace/LLM) and [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model)
  - [FP8/FP6/FP4 inference](python/llm/example/GPU/HuggingFace/More-Data-Types): **FP8**, **FP6** and **FP4** LLM inference on Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types)
  - [INT8 inference](python/llm/example/GPU/HuggingFace/More-Data-Types): **INT8** LLM inference on Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types) and [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types)
  - [INT2 inference](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2): **INT2** LLM inference (based on llama.cpp IQ2 mechanism) on Intel [GPU](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2)
- #### FP16/BF16 inference
  - **FP16** LLM inference on Intel [GPU](python/llm/example/GPU/Speculative-Decoding), with possible [self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md) optimization
  - **BF16** LLM inference on Intel [CPU](python/llm/example/CPU/Speculative-Decoding), with possible [self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md) optimization
- #### Distributed inference
  - **Pipeline Parallel** inference on Intel [GPU](python/llm/example/GPU/Pipeline-Parallel-Inference)
  - **DeepSpeed AutoTP** inference on Intel [GPU](python/llm/example/GPU/Deepspeed-AutoTP)
- #### Save and load
  - [Low-bit models](python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load): saving and loading `ipex-llm` low-bit models (INT4/FP4/FP6/INT8/FP8/FP16/etc.)
  - [GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF): directly loading GGUF models into `ipex-llm`
  - [AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ): directly loading AWQ models into `ipex-llm`
  - [GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ): directly loading GPTQ models into `ipex-llm`
- #### Finetuning
  - LLM finetuning on Intel [GPU](python/llm/example/GPU/LLM-Finetuning), including [LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) and [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)
  - QLoRA finetuning on Intel [CPU](python/llm/example/CPU/QLoRA-FineTuning)
- #### Integration with community libraries
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
- [Tutorials](https://github.com/intel-analytics/ipex-llm-tutorial)

## API Doc
- [HuggingFace Transformers-style API (Auto Classes)](docs/mddocs/PythonAPI/transformers.md)
- [API for arbitrary PyTorch Model](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/PythonAPI/optimize.md)

## FAQ
- [FAQ & Trouble Shooting](docs/mddocs/Overview/FAQ/faq.md)

## Verified Models
Over 50 models have been optimized/verified on `ipex-llm`, including *LLaMA/LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM2/ChatGLM3, Baichuan/Baichuan2, Qwen/Qwen-1.5, InternLM* and more; see the list below.
  
| Model      | CPU Example                                                    | GPU Example                                                     |
|------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](python/llm/example/GPU/HuggingFace/LLM/vicuna)|
| LLaMA 2    | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link](python/llm/example/GPU/HuggingFace/LLM/llama2)  |
| LLaMA 3    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3)  |
| LLaMA 3.1    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3.1) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3.1)  |
| ChatGLM    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |    | 
| ChatGLM2   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm2)   |
| ChatGLM3   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm3)   |
| GLM-4      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm4)      | [link](python/llm/example/GPU/HuggingFace/LLM/glm4)       |
| GLM-4V     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm-4v)    | [link](python/llm/example/GPU/HuggingFace/Multimodal/glm-4v)     |
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
| Baichuan2  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](python/llm/example/GPU/HuggingFace/LLM/baichuan2)  |
| InternLM   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm)  | [link](python/llm/example/GPU/HuggingFace/LLM/internlm)   |
| Qwen       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen)      | [link](python/llm/example/GPU/HuggingFace/LLM/qwen)       |
| Qwen1.5 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen1.5) |
| Qwen2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen2) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen2) |
| Qwen-VL    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen-vl)   | [link](python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl)    |
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
| MiniCPM | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/minicpm) | [link](python/llm/example/GPU/HuggingFace/LLM/minicpm) |
| MiniCPM-V |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V) |
| MiniCPM-V-2 |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2) |
| MiniCPM-Llama3-V-2_5 |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-Llama3-V-2_5) |
| MiniCPM-V-2_6 |  | [link](python/llm/example/GPU/HuggingFace/Multimodal/MiniCPM-V-2_6) | 

## Get Support
- Please report a bug or raise a feature request by opening a [Github Issue](https://github.com/intel-analytics/ipex-llm/issues)
- Please report a vulnerability by opening a draft [GitHub Security Advisory](https://github.com/intel-analytics/ipex-llm/security/advisories)
