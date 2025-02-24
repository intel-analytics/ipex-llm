#  💫 Intel® LLM Library for PyTorch* 
<p>
  <b>< English</b> | <a href='./README.zh-CN.md'>中文</a> >
</p>

**`IPEX-LLM`** is an LLM acceleration library for Intel [GPU](Quickstart/install_windows_gpu.md) *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*, [NPU](Quickstart/npu_quickstart.md) and CPU [^1].

## Latest Update 🔥 
- [2025/02] We added support of [Ollama Portable Zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run Ollama on Intel GPU for both [Windows](Quickstart/ollama_portable_zip_quickstart.md#windows-quickstart) and [Linux](docs/mddocs/Quickstart/ollama_portable_zip_quickstart.md#linux-quickstart) (***without the need of manual installations***).
- [2025/02] We added support for running [vLLM 0.6.6](DockerGuides/vllm_docker_quickstart.md) on Intel Arc GPUs.
- [2025/01] We added the guide for running `ipex-llm` on Intel Arc [B580](Quickstart/bmg_quickstart.md) GPU
- [2025/01] We added support for running [Ollama 0.5.4](Quickstart/ollama_quickstart.md) on Intel GPU.
- [2024/12] We added both ***Python*** and ***C++*** support for Intel Core Ultra [NPU](Quickstart/npu_quickstart.md) (including 100H, 200V and 200K series).

<details><summary>More updates</summary>
<br/>

- [2024/11] We added support for running [vLLM 0.6.2](DockerGuides/vllm_docker_quickstart.md) on Intel Arc GPUs.
- [2024/07] We added support for running Microsoft's **GraphRAG** using local LLM on Intel GPU; see the quickstart guide [here](Quickstart/graphrag_quickstart.md).
- [2024/07] We added extensive support for Large Multimodal Models, including [StableDiffusion](../../python/llm/example/GPU/HuggingFace/Multimodal/StableDiffusion), [Phi-3-Vision](../../python/llm/example/GPU/HuggingFace/Multimodal/phi-3-vision), [Qwen-VL](../../python/llm/example/GPU/HuggingFace/Multimodal/qwen-vl), and [more](../../python/llm/example/GPU/HuggingFace/Multimodal).
- [2024/07] We added **FP6** support on Intel [GPU](../../python/llm/example/GPU/HuggingFace/More-Data-Types). 
- [2024/06] We added experimental **NPU** support for Intel Core Ultra processors; see the examples [here](../../python/llm/example/NPU/HF-Transformers-AutoModels). 
- [2024/06] We added extensive support of **pipeline parallel** [inference](../../python/llm/example/GPU/Pipeline-Parallel-Inference), which makes it easy to run large-sized LLM using 2 or more Intel GPUs (such as Arc).
- [2024/06] We added support for running **RAGFlow** with `ipex-llm` on Intel [GPU](Quickstart/ragflow_quickstart.md).
- [2024/05] `ipex-llm` now supports **Axolotl** for LLM finetuning on Intel GPU; see the quickstart [here](Quickstart/axolotl_quickstart.md). 
- [2024/05] You can now easily run `ipex-llm` inference, serving and finetuning using the **Docker** [images](#docker).
- [2024/05] You can now install `ipex-llm` on Windows using just "*[one command](Quickstart/install_windows_gpu.md#install-ipex-llm)*".
- [2024/04] You can now run **Open WebUI** on Intel GPU using `ipex-llm`; see the quickstart [here](Quickstart/open_webui_with_ollama_quickstart.md).
- [2024/04] You can now run **Llama 3** on Intel GPU using `llama.cpp` and `ollama` with `ipex-llm`; see the quickstart [here](Quickstart/llama3_llamacpp_ollama_quickstart.md).
- [2024/04] `ipex-llm` now supports **Llama 3** on both Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM/llama3) and [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3).
- [2024/04] `ipex-llm` now provides C++ interface, which can be used as an accelerated backend for running [llama.cpp](Quickstart/llama_cpp_quickstart.md) and [ollama](Quickstart/ollama_quickstart.md) on Intel GPU.
- [2024/03] `bigdl-llm` has now become `ipex-llm` (see the migration guide [here](Quickstart/bigdl_llm_migration.md)); you may find the original `BigDL` project [here](https://github.com/intel-analytics/bigdl-2.x).
- [2024/02] `ipex-llm` now supports directly loading model from [ModelScope](../../python/llm/example/GPU/ModelScope-Models) ([魔搭](../../python/llm/example/CPU/ModelScope-Models)).
- [2024/02] `ipex-llm` added initial **INT2** support (based on llama.cpp [IQ2](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2) mechanism), which makes it possible to run large-sized LLM (e.g., Mixtral-8x7B) on Intel GPU with 16GB VRAM.
- [2024/02] Users can now use `ipex-llm` through [Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUI.
- [2024/02] `ipex-llm` now supports *[Self-Speculative Decoding](Inference/Self_Speculative_Decoding.md)*, which in practice brings **~30% speedup** for FP16 and BF16 inference latency on Intel [GPU](../../python/llm/example/GPU/Speculative-Decoding) and [CPU](../../python/llm/example/CPU/Speculative-Decoding) respectively.
- [2024/02] `ipex-llm` now supports a comprehensive list of LLM **finetuning** on Intel GPU (including [LoRA](../../python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](../../python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) and [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora)).
- [2024/01] Using `ipex-llm` [QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA), we managed to finetune LLaMA2-7B in **21 minutes** and LLaMA2-70B in **3.14 hours** on 8 Intel Max 1550 GPU for [Standford-Alpaca](../../python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora) (see the blog [here](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)). 
- [2023/12] `ipex-llm` now supports [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora) (see *["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*).
- [2023/12] `ipex-llm` now supports [Mixtral-8x7B](../../python/llm/example/GPU/HuggingFace/LLM/mixtral) on both Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM/mixtral) and [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral). 
- [2023/12] `ipex-llm` now supports [QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) (see *["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*). 
- [2023/12] `ipex-llm` now supports [FP8 and FP4 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types) on Intel ***GPU***.
- [2023/11] Initial support for directly loading [GGUF](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF), [AWQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ) and [GPTQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ) models into `ipex-llm` is available.
- [2023/11] `ipex-llm` now supports [vLLM continuous batching](../../python/llm/example/GPU/vLLM-Serving) on both Intel [GPU](../../python/llm/example/GPU/vLLM-Serving) and [CPU](../../python/llm/example/CPU/vLLM-Serving).
- [2023/10] `ipex-llm` now supports [QLoRA finetuning](../../python/llm/example/GPU/LLM-Finetuning/QLoRA) on both Intel [GPU](../../python/llm/example/GPU/LLM-Finetuning/QLoRA) and [CPU](../../python/llm/example/CPU/QLoRA-FineTuning).
- [2023/10] `ipex-llm` now supports [FastChat serving](../../python/llm/src/ipex_llm/llm/serving) on on both Intel CPU and GPU.
- [2023/09] `ipex-llm` now supports [Intel GPU](../../python/llm/example/GPU) (including iGPU, Arc, Flex and MAX).
- [2023/09] `ipex-llm` [tutorial](https://github.com/intel-analytics/ipex-llm-tutorial) is released.
 
</details> 

## `ipex-llm` Quickstart

### Use
- [Ollama Portable Zip](Quickstart/ollama_portable_zip_quickstart.md): running **Ollama** on Intel GPU ***without the need of manual installations***
- [Arc B580](Quickstart/bmg_quickstart.md): running `ipex-llm` on Intel Arc **B580** GPU for Ollama, llama.cpp, PyTorch, HuggingFace, etc.
- [NPU](Quickstart/npu_quickstart.md): running `ipex-llm` on Intel **NPU** in both Python and C++
- [llama.cpp](Quickstart/llama_cpp_quickstart.md): running **llama.cpp** (*using C++ interface of `ipex-llm`*) on Intel GPU
- [Ollama](Quickstart/ollama_quickstart.md): running **ollama** (*using C++ interface of `ipex-llm`*) on Intel GPU
- [PyTorch/HuggingFace](Quickstart/install_windows_gpu.md): running **PyTorch**, **HuggingFace**, **LangChain**, **LlamaIndex**, etc. (*using Python interface of `ipex-llm`*) on Intel GPU for [Windows](Quickstart/install_windows_gpu.md) and [Linux](Quickstart/install_linux_gpu.md)
- [vLLM](Quickstart/vLLM_quickstart.md): running `ipex-llm` in **vLLM** on both Intel [GPU](DockerGuides/vllm_docker_quickstart.md) and [CPU](DockerGuides/vllm_cpu_docker_quickstart.md)
- [FastChat](Quickstart/fastchat_quickstart.md): running `ipex-llm` in **FastChat** serving on on both Intel GPU and CPU
- [Serving on multiple Intel GPUs](Quickstart/deepspeed_autotp_fastapi_quickstart.md): running `ipex-llm` **serving on multiple Intel GPUs** by leveraging DeepSpeed AutoTP and FastAPI
- [Text-Generation-WebUI](Quickstart/webui_quickstart.md): running `ipex-llm` in `oobabooga` **WebUI**
- [Axolotl](Quickstart/axolotl_quickstart.md): running `ipex-llm` in **Axolotl** for LLM finetuning
- [Benchmarking](Quickstart/benchmark_quickstart.md): running  (latency and throughput) **benchmarks** for `ipex-llm` on Intel CPU and GPU

### Docker
- [GPU Inference in C++](DockerGuides/docker_cpp_xpu_quickstart.md): running `llama.cpp`, `ollama`, etc., with `ipex-llm` on Intel GPU
- [GPU Inference in Python](DockerGuides/docker_pytorch_inference_gpu.md) : running HuggingFace `transformers`, `LangChain`, `LlamaIndex`, `ModelScope`, etc. with `ipex-llm` on Intel GPU
- [vLLM on GPU](DockerGuides/vllm_docker_quickstart.md): running `vLLM` serving with `ipex-llm` on Intel GPU
- [vLLM on CPU](DockerGuides/vllm_cpu_docker_quickstart.md): running `vLLM` serving with `ipex-llm` on Intel CPU  
- [FastChat on GPU](DockerGuides/fastchat_docker_quickstart.md): running `FastChat` serving with `ipex-llm` on Intel GPU
- [VSCode on GPU](DockerGuides/docker_run_pytorch_inference_in_vscode.md): running and developing `ipex-llm` applications in Python using VSCode on Intel GPU

### Applications
- [GraphRAG](Quickstart/graphrag_quickstart.md): running Microsoft's `GraphRAG` using local LLM with `ipex-llm`
- [RAGFlow](Quickstart/ragflow_quickstart.md): running `RAGFlow` (*an open-source RAG engine*) with `ipex-llm`
- [LangChain-Chatchat](Quickstart/chatchat_quickstart.md): running `LangChain-Chatchat` (*Knowledge Base QA using RAG pipeline*) with `ipex-llm`
- [Coding copilot](Quickstart/continue_quickstart.md): running `Continue` (coding copilot in VSCode) with `ipex-llm`
- [Open WebUI](Quickstart/open_webui_with_ollama_quickstart.md): running `Open WebUI` with `ipex-llm`
- [PrivateGPT](Quickstart/privateGPT_quickstart.md): running `PrivateGPT` to interact with documents with `ipex-llm`
- [Dify platform](Quickstart/dify_quickstart.md): running `ipex-llm` in `Dify`(*production-ready LLM app development platform*)

### Install 
- [Windows GPU](Quickstart/install_windows_gpu.md): installing `ipex-llm` on Windows with Intel GPU
- [Linux GPU](Quickstart/install_linux_gpu.md): installing `ipex-llm` on Linux with Intel GPU
- *For more details, please refer to the [full installation guide](Overview/install.md)*

### Code Examples
- #### Low bit inference
  - [INT4 inference](../../python/llm/example/GPU/HuggingFace/LLM): **INT4** LLM inference on Intel [GPU](../../python/llm/example/GPU/HuggingFace/LLM) and [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/Model)
  - [FP8/FP6/FP4 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types): **FP8**, **FP6** and **FP4** LLM inference on Intel [GPU](../../python/llm/example/GPU/HuggingFace/More-Data-Types)
  - [INT8 inference](../../python/llm/example/GPU/HuggingFace/More-Data-Types): **INT8** LLM inference on Intel [GPU](../../python/llm/example/GPU/HuggingFace/More-Data-Types) and [CPU](../../python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types)
  - [INT2 inference](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2): **INT2** LLM inference (based on llama.cpp IQ2 mechanism) on Intel [GPU](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2)
- #### FP16/BF16 inference
  - **FP16** LLM inference on Intel [GPU](../../python/llm/example/GPU/Speculative-Decoding), with possible [self-speculative decoding](Inference/Self_Speculative_Decoding.md) optimization
  - **BF16** LLM inference on Intel [CPU](../../python/llm/example/CPU/Speculative-Decoding), with possible [self-speculative decoding](Inference/Self_Speculative_Decoding.md) optimization
- #### Distributed inference
  - **Pipeline Parallel** inference on Intel [GPU](../../python/llm/example/GPU/Pipeline-Parallel-Inference)
  - **DeepSpeed AutoTP** inference on Intel [GPU](../../python/llm/example/GPU/Deepspeed-AutoTP)
- #### Save and load
  - [Low-bit models](../../python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load): saving and loading `ipex-llm` low-bit models (INT4/FP4/FP6/INT8/FP8/FP16/etc.)
  - [GGUF](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF): directly loading GGUF models into `ipex-llm`
  - [AWQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ): directly loading AWQ models into `ipex-llm`
  - [GPTQ](../../python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ): directly loading GPTQ models into `ipex-llm`
- #### Finetuning
  - LLM finetuning on Intel [GPU](../../python/llm/example/GPU/LLM-Finetuning), including [LoRA](../../python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](../../python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](../../python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](../../python/llm/example/GPU/LLM-Finetuning/QA-LoRA) and [ReLoRA](../../python/llm/example/GPU/LLM-Finetuning/ReLora)
  - QLoRA finetuning on Intel [CPU](../../python/llm/example/CPU/QLoRA-FineTuning)
- #### Integration with community libraries
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
- [Tutorials](https://github.com/intel-analytics/ipex-llm-tutorial)

## API Doc
- [HuggingFace Transformers-style API (Auto Classes)](PythonAPI/transformers.md)
- [API for arbitrary PyTorch Model](https://github.com/intel-analytics/ipex-llm/blob/main/PythonAPI/optimize.md)

## FAQ
- [FAQ & Trouble Shooting](Overview/FAQ/faq.md)

## Verified Models
Over 70 models have been optimized/verified on `ipex-llm`, including *LLaMA/LLaMA2, Mistral, Mixtral, Gemma, LLaVA, Whisper, ChatGLM2/ChatGLM3, Baichuan/Baichuan2, Qwen/Qwen-1.5, InternLM* and more; see the list below.
  
| Model      | CPU Example                                  | GPU Example                                  | NPU Example                                  |
|------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|
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
