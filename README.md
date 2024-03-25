## IPEX-LLM

**`ipex-llm`** is a library for running **LLM** (large language model) on Intel **XPU** (from *Laptop* to *GPU* to *Cloud*) using **INT4/FP4/INT8/FP8** with very low latency[^1] (for any **PyTorch** model).

> *It is built on the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [qlora](https://github.com/artidoro/qlora), [gptq](https://github.com/IST-DASLab/gptq), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [awq](https://github.com/mit-han-lab/llm-awq), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [vLLM](https://github.com/vllm-project/vllm), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.*

### Latest update 🔥 
- [2024/03] **LangChain** added support for `ipex-llm`; see the details [here](https://python.langchain.com/docs/integrations/llms/bigdl).
- [2024/02] `ipex-llm` now supports directly loading model from [ModelScope](python/llm/example/GPU/ModelScope-Models) ([魔搭](python/llm/example/CPU/ModelScope-Models)).
- [2024/02] `ipex-llm` added inital **INT2** support (based on llama.cpp [IQ2](python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF-IQ2) mechanism), which makes it possible to run large-size LLM (e.g., Mixtral-8x7B) on Intel GPU with 16GB VRAM.
- [2024/02] Users can now use `ipex-llm` through [Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUI.
- [2024/02] `ipex-llm` now supports *[Self-Speculative Decoding](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Inference/Self_Speculative_Decoding.html)*, which in practice brings **~30% speedup** for FP16 and BF16 inference latency on Intel [GPU](python/llm/example/GPU/Speculative-Decoding) and [CPU](python/llm/example/CPU/Speculative-Decoding) respectively.
- [2024/02] `ipex-llm` now supports a comprehensive list of LLM finetuning on Intel GPU (including [LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA), [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), [DPO](python/llm/example/GPU/LLM-Finetuning/DPO), [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) and [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)).
- [2024/01] Using `ipex-llm` [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA), we managed to finetune LLaMA2-7B in **21 minutes** and LLaMA2-70B in **3.14 hours** on 8 Intel Max 1550 GPU for [Standford-Alpaca](python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora) (see the blog [here](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)).
- [2024/01] 🔔🔔🔔 ***The default `ipex-llm` GPU Linux installation has switched from PyTorch 2.0 to PyTorch 2.1, which requires new oneAPI and GPU driver versions. (See the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more details.)***
- [2023/12] `ipex-llm` now supports [ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora) (see *["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*).
- [2023/12] `ipex-llm` now supports [Mixtral-8x7B](python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral) on both Intel [GPU](python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral) and [CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral).
- [2023/12] `ipex-llm` now supports [QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA) (see *["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*).
- [2023/12] `ipex-llm` now supports [FP8 and FP4 inference](python/llm/example/GPU/HF-Transformers-AutoModels/More-Data-Types) on Intel ***GPU***.
- [2023/11] Initial support for directly loading [GGUF](python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GGUF), [AWQ](python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/AWQ) and [GPTQ](python/llm/example/GPU/HF-Transformers-AutoModels/Advanced-Quantizations/GPTQ) models into `ipex-llm` is available.
- [2023/11] `ipex-llm` now supports [vLLM continuous batching](python/llm/example/GPU/vLLM-Serving) on both Intel [GPU](python/llm/example/GPU/vLLM-Serving) and [CPU](python/llm/example/CPU/vLLM-Serving).
- [2023/10] `ipex-llm` now supports [QLoRA finetuning](python/llm/example/GPU/LLM-Finetuning/QLoRA) on both Intel [GPU](python/llm/example/GPU/LLM-Finetuning/QLoRA) and [CPU](python/llm/example/CPU/QLoRA-FineTuning).
- [2023/10] `ipex-llm` now supports [FastChat serving](python/llm/src/ipex_llm/llm/serving) on on both Intel CPU and GPU.
- [2023/09] `ipex-llm` now supports [Intel GPU](python/llm/example/GPU) (including iGPU, Arc, Flex and MAX).
- [2023/09] `ipex-llm` [tutorial](https://github.com/intel-analytics/ipex-llm-tutorial) is released.
- [2023/09] Over 40 models have been optimized/verified on `ipex-llm`, including *LLaMA/LLaMA2, ChatGLM2/ChatGLM3, Mistral, Falcon, MPT, LLaVA, WizardCoder, Dolly, Whisper, Baichuan/Baichuan2, InternLM, Skywork, QWen/Qwen-VL, Aquila, MOSS,* and more; see the complete list [here](#verified-models).

### `ipex-llm` Demos
See the ***optimized performance*** of `chatglm2-6b` and `llama-2-13b-chat` models on 12th Gen Intel Core CPU and Intel Arc GPU below.

<table width="100%">
  <tr>
    <td align="center" colspan="2">12th Gen Intel Core CPU</td>
    <td align="center" colspan="2">Intel Arc GPU</td>
  </tr>
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" ></a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif"></a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif"></a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif"></a>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%"><code>chatglm2-6b</code></td>
    <td align="center" width="25%"><code>llama-2-13b-chat</code></td>
    <td align="center" width="25%"><code>chatglm2-6b</code></td>
    <td align="center" width="25%"><code>llama-2-13b-chat</code></td>
  </tr>
</table>

### `ipex-llm` quickstart

- [Windows GPU installation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html)
- [Run IPEX-LLM in Text-Generation-WebUI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html)
- [Run IPEX-LLM using Docker](docker/llm)
- [CPU INT4](#cpu-int4)
- [GPU INT4](#gpu-int4)
- [More Low-Bit support](#more-low-bit-support)
- [Verified models](#verified-models)

#### CPU INT4
##### Install
You may install **`ipex-llm`** on Intel CPU as follows:
> Note: See the [CPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_cpu.html) for more details.
```bash
pip install --pre --upgrade ipex-llm[all]
```
> Note: `ipex-llm` has been tested on Python 3.9, 3.10 and 3.11

##### Run Model
You may apply INT4 optimizations to any Hugging Face *Transformers* models as follows.

```python
#load Hugging Face Transformers model with INT4 optimizations
from ipex_llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

#run the optimized model on CPU
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...)
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids)
```
*See the complete examples [here](python/llm/example/CPU/HF-Transformers-AutoModels/Model).*

#### GPU INT4
##### Install
You may install **`ipex-llm`** on Intel GPU as follows:
> Note: See the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more details.
```bash
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
> Note: `ipex-llm` has been tested on Python 3.9, 3.10 and 3.11

##### Run Model
You may apply INT4 optimizations to any Hugging Face *Transformers* models as follows.

```python
#load Hugging Face Transformers model with INT4 optimizations
from ipex_llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

#run the optimized model on Intel GPU
model = model.to('xpu')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...).to('xpu')
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids.cpu())
```
*See the complete examples [here](python/llm/example/GPU).*

#### More Low-Bit Support
##### Save and load

After the model is optimized using `ipex-llm`, you may save and load the model as follows:
```python
model.save_low_bit(model_path)
new_model = AutoModelForCausalLM.load_low_bit(model_path)
```
*See the complete example [here](python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load).*

##### Additonal data types

In addition to INT4, You may apply other low bit optimizations (such as *INT8*, *INT5*, *NF4*, etc.) as follows: 
```python
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_low_bit="sym_int8")
```
*See the complete example [here](python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types).*

#### Verified Models
Over 40 models have been optimized/verified on `ipex-llm`, including *LLaMA/LLaMA2, ChatGLM/ChatGLM2, Mistral, Falcon, MPT, Baichuan/Baichuan2, InternLM, QWen* and more; see the example list below.

| Model                                    | CPU Example                              | GPU Example                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/vicuna) |
| LLaMA 2                                  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link1](python/llm/example/GPU/HF-Transformers-AutoModels/Model/llama2), [link2-low GPU memory example](python/llm/example/GPU/PyTorch-Models/Model/llama2#example-2---low-memory-version-predict-tokens-using-generate-api) |
| ChatGLM                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm) |                                          |
| ChatGLM2                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/chatglm2) |
| ChatGLM3                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/chatglm3) |
| Mistral                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/mistral) |
| Mixtral                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/mixtral) |
| Falcon                                   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/falcon) |
| MPT                                      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/mpt) |
| Dolly-v1                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/dolly-v1) |
| Dolly-v2                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/dolly-v2) |
| Replit Code                              | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/replit) |
| RedPajama                                | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama) |                                          |
| Phoenix                                  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix) |                                          |
| StarCoder                                | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/starcoder) |
| Baichuan                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/baichuan) |
| Baichuan2                                | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/baichuan2) |
| InternLM                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/internlm) |
| Qwen                                     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen) |
| Qwen1.5                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen1.5) |
| Qwen-VL                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen-vl) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/qwen-vl) |
| Aquila                                   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/aquila) |
| Aquila2                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/aquila2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/aquila2) |
| MOSS                                     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss) |                                          |
| Whisper                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/whisper) |
| Phi-1_5                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-1_5) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/phi-1_5) |
| Flan-t5                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/flan-t5) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/flan-t5) |
| LLaVA                                    | [link](python/llm/example/CPU/PyTorch-Models/Model/llava) | [link](python/llm/example/GPU/PyTorch-Models/Model/llava) |
| CodeLlama                                | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codellama) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/codellama) |
| Skywork                                  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/skywork) |                                          |
| InternLM-XComposer                       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm-xcomposer) |                                          |
| WizardCoder-Python                       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/wizardcoder-python) |                                          |
| CodeShell                                | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/codeshell) |                                          |
| Fuyu                                     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/fuyu) |                                          |
| Distil-Whisper                           | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/distil-whisper) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/distil-whisper) |
| Yi                                       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/yi) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/yi) |
| BlueLM                                   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/bluelm) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/bluelm) |
| Mamba                                    | [link](python/llm/example/CPU/PyTorch-Models/Model/mamba) | [link](python/llm/example/GPU/PyTorch-Models/Model/mamba) |
| SOLAR                                    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/solar) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/solar) |
| Phixtral                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phixtral) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/phixtral) |
| InternLM2                                | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/internlm2) |
| RWKV4                                    |                                          | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/rwkv4) |
| RWKV5                                    |                                          | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/rwkv5) |
| Bark                                     | [link](python/llm/example/CPU/PyTorch-Models/Model/bark) | [link](python/llm/example/GPU/PyTorch-Models/Model/bark) |
| SpeechT5                                 |                                          | [link](python/llm/example/GPU/PyTorch-Models/Model/speech-t5) |
| DeepSeek-MoE                             | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek-moe) |                                          |
| Ziya-Coding-34B-v1.0                     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/ziya) |                                          |
| Phi-2                                    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phi-2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/phi-2) |
| Yuan2                                    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/yuan2) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/yuan2) |
| Gemma                                    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/gemma) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/gemma) |
| DeciLM-7B                                | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deciLM-7b) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/deciLM-7b) |
| Deepseek                                 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/deepseek) | [link](python/llm/example/GPU/HF-Transformers-AutoModels/Model/deepseek) |


***For more details, please refer to the `ipex-llm` [Document](https://test-ipex-llm.readthedocs.io/en/main/doc/LLM/index.html), [Readme](python/llm), [Tutorial](https://github.com/intel-analytics/ipex-llm-tutorial) and [API Doc](https://ipex-llm.readthedocs.io/en/latest/doc/PythonAPI/LLM/index.html).***
