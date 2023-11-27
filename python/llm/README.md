## BigDL-LLM


**[BigDL-LLM](https://bigdl.readthedocs.io/en/latest/doc/LLM/index.html)** [^2] is a library designed for running large language models (LLMs) efficiently on Intel XPU, ranging from laptops to GPUs to cloud environments. It leverages low-bit (INT4/INT5/INT8,etc.) optimizations to achieve low-latency performance[^1] across various PyTorch models.

### Key Features

- Optimized performance and low memory footprint for a wide range of Intel XPU (Xeon/Core/Flex/Arc/iGPU)
- Various low-bit support: **INT4**/INT5/INT8/FP4/NF4/FP8, etc
- Allows for both inference and fine-tune on XPU
- Easy-to-use APIs (HF transformers, langchain, etc.)
- Verfied on 30+ models with abundant examples

### Quick Links
- [API Documentation](#api-documentation)
- [List of Verified Models/Examples](#verified-models)
- [Try BigDL-LLM without Installation](#portable-zip-run-bigdl-llm-without-installation)
- [Installation Guide](#installation)
- [Usage Examples](#usage-examples)


### Demos
Explore the optimized performance of models like `chatglm2-6b` and `llama-2-13b-chat` on 12th Gen Intel Core CPU and Intel Arc GPU. Click on the images below to see these models in action:

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

### Verified Models & Examples

BigDL-LLM can be applied to transformer-based models. Here's a list of verified models along with examples for CPU and GPU usage:
  
| Model      | CPU Example                                                    | GPU Example                                                     |
|------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](example/GPU/HF-Transformers-AutoModels/Model/vicuna)|
| LLaMA 2    | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link](example/GPU/HF-Transformers-AutoModels/Model/llama2) |
| ChatGLM    | [link](example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |    | 
| ChatGLM2   | [link](example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  | [link](example/GPU/HF-Transformers-AutoModels/Model/chatglm2)   |
| ChatGLM3   | [link](example/CPU/HF-Transformers-AutoModels/Model/chatglm3)  | [link](example/GPU/HF-Transformers-AutoModels/Model/chatglm3)   |
| Mistral    | [link](example/CPU/HF-Transformers-AutoModels/Model/mistral)   | [link](example/GPU/HF-Transformers-AutoModels/Model/mistral)    |
| Falcon     | [link](example/CPU/HF-Transformers-AutoModels/Model/falcon)    | [link](example/GPU/HF-Transformers-AutoModels/Model/falcon)     |
| MPT        | [link](example/CPU/HF-Transformers-AutoModels/Model/mpt)       | [link](example/CPU/HF-Transformers-AutoModels/Model/mpt)        |
| Dolly-v1   | [link](example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)  | [link](example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)   | 
| Dolly-v2   | [link](example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)  | [link](example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)   | 
| Replit Code| [link](example/CPU/HF-Transformers-AutoModels/Model/replit)    | [link](example/CPU/HF-Transformers-AutoModels/Model/replit)     |
| RedPajama  | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/redpajama) |    | 
| Phoenix    | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/phoenix)   |    | 
| StarCoder  | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/starcoder) | [link](example/GPU/HF-Transformers-AutoModels/Model/starcoder) | 
| Baichuan   | [link](example/CPU/HF-Transformers-AutoModels/Model/baichuan)  | [link](example/CPU/HF-Transformers-AutoModels/Model/baichuan)   |
| Baichuan2  | [link](example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](example/GPU/HF-Transformers-AutoModels/Model/baichuan2)  |
| InternLM   | [link](example/CPU/HF-Transformers-AutoModels/Model/internlm)  | [link](example/GPU/HF-Transformers-AutoModels/Model/internlm)   |
| Qwen       | [link](example/CPU/HF-Transformers-AutoModels/Model/qwen)      | [link](example/GPU/HF-Transformers-AutoModels/Model/qwen)       | 
| Qwen-VL    | [link](example/CPU/HF-Transformers-AutoModels/Model/qwen-vl)   | [link](example/GPU/HF-Transformers-AutoModels/Model/qwen-vl)    |
| Aquila     | [link](example/CPU/HF-Transformers-AutoModels/Model/aquila)    | [link](example/GPU/HF-Transformers-AutoModels/Model/aquila)     |
| Aquila2    | [link](example/CPU/HF-Transformers-AutoModels/Model/aquila2)   | [link](example/GPU/HF-Transformers-AutoModels/Model/aquila2)    |
| MOSS       | [link](example/CPU/HF-Transformers-AutoModels/Model/moss)      |    | 
| Whisper    | [link](example/CPU/HF-Transformers-AutoModels/Model/whisper)   | [link](example/GPU/HF-Transformers-AutoModels/Model/whisper)    |
| Phi-1_5    | [link](example/CPU/HF-Transformers-AutoModels/Model/phi-1_5)   | [link](example/GPU/HF-Transformers-AutoModels/Model/phi-1_5)    |
| Flan-t5    | [link](example/CPU/HF-Transformers-AutoModels/Model/flan-t5)   | [link](example/GPU/HF-Transformers-AutoModels/Model/flan-t5)    |
| Qwen-VL    | [link](example/CPU/HF-Transformers-AutoModels/Model/qwen-vl)   |   |
| LLaVA      | [link](example/CPU/PyTorch-Models/Model/llava)                 | [link](example/GPU/PyTorch-Models/Model/llava)                  |
| CodeLlama  | [link](example/CPU/HF-Transformers-AutoModels/Model/codellama) | [link](example/GPU/HF-Transformers-AutoModels/Model/codellama)  |
| Skywork    | [link](example/CPU/HF-Transformers-AutoModels/Model/skywork)                 |    |
| InternLM-XComposer    | [link](example/CPU/HF-Transformers-AutoModels/Model/internlm-xcomposer)   |   |
| WizardCoder-Python | [link](example/CPU/HF-Transformers-AutoModels/Model/wizardcoder-python) | |
| CodeShell | [link](example/CPU/HF-Transformers-AutoModels/Model/codeshell) | |
| Fuyu      | [link](example/CPU/HF-Transformers-AutoModels/Model/fuyu) | |
| Distil-Whisper | [link](example/CPU/HF-Transformers-AutoModels/Model/distil-whisper) | [link](example/GPU/HF-Transformers-AutoModels/Model/distil-whisper) |
| Yi | [link](example/CPU/HF-Transformers-AutoModels/Model/yi) | [link](example/GPU/HF-Transformers-AutoModels/Model/yi) |

### Installation

#### For Intel CPU

```bash
# Note: Tested on Python 3.9
pip install --pre --upgrade bigdl-llm[all]
```

#### For Intel GPU
```bash
# Note: Tested on Python 3.9
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### Portable Zip: Run BigDL-LLM Without Installation

[TODO]

### Usage Examples

BigDL-LLM offers multiple ways to work with large language models. Here's a guide to help you get started:

#### 1. Hugging Face `transformers` API
BigDL-LLM seamlessly integrates with the Hugging Face [transformers API](), allowing you to run and optimize Transformer models with ease.

* **Inference w/ INT4 on Intel CPU**: Enhance your models' performance on CPUs using INT4 optimizations for faster inference with minimal accuracy loss.

  ```python
  #load Hugging Face Transformers model with INT4 optimizations
  from bigdl.llm.transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained(model_id_or_path, load_in_4bit=True) 
  # Additional example code...
  ```
  [View detailed CPU examples](example/CPU/HF-Transformers-AutoModels/Model/)


* **Inference w/ INT4 on Intel GPU**: Similarly, optimize your models for Intel GPUs with INT4 optimizations.
  ```python
  #load Hugging Face Transformers model with INT4 optimizations
  from bigdl.llm.transformers import AutoModelForCausalLM
  import intel_extension_for_pytorch
  model = AutoModelForCausalLM.from_pretrained(model_id_or_path, load_in_4bit=True) 
  model = model.to('xpu')
  # Additional example code...
  ```
  [View detailed GPU examples](example/GPU/HF-Transformers-AutoModels/Model)

* **Save & Load**: After the model is optimized, you may save and load the model for future use.
  ```python
  from bigdl.llm.transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained(model_id_or_path, load_in_4bit=True)
  #save optimized model
  model.save_low_bit(optimized_model_path)
  # load optimized model 
  loaded_model = AutoModelForCausalLM.load_low_bit(optimized_model_path)
  ```
  [View detailed examples](example/CPU/HF-Transformers-AutoModels/Save-Load) 

- **Other Data Types**: In addition to INT4, You may apply other low bit optimizations (such as *INT8*, *INT5*, *NF4*, etc.).

  ```python
  model = AutoModelForCausalLM.from_pretrained(model_id_or_path, load_in_low_bit="sym_int8") # or `asym_int8`, `sym_int5`, `asym_int5`, `nf4`, etc. 
  ```
  [View detailed example](example/CPU/HF-Transformers-AutoModels/More-Data-Types)

#### 2. PyTorch API
[TODO]

#### 3. LangChain API
Utilize LangChain with BigDL-LLM to enhance your model's interactivity and contextual understanding for tasks like retrieval augmented QA, multi-turn conversation with memory, etc.

  ```python
  from bigdl.llm.langchain.llms import TransformersLLM
  bigdl_llm = TransformersLLM.from_model_id(model_id=model_path, ...)
  # Additional example code...
  ```
  [Explore LangChain integration examples](example/CPU/LangChain/transformers_int4).
 

#### 4. Fine-Tune w/ peft & transformers API

[TODO]

### API Documentation
For a comprehensive guide and detailed documentation on BigDL-LLM's features and capabilities, visit our [API Documentation](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/LLM/index.html).

### Compatibility and Dependency

The native code/lib in `bigdl-llm` has been built using the following tools.
Note that lower  `LIBC` version on your Linux system may be incompatible with `bigdl-llm`.

| Model family | Platform | Compiler           | GLIBC |
| ------------ | -------- | ------------------ | ----- |
| llama        | Linux    | GCC 11.2.1         | 2.17  |
| llama        | Windows  | MSVC 19.36.32532.0 |       |
| llama        | Windows  | GCC 13.1.0         |       |
| gptneox      | Linux    | GCC 11.2.1         | 2.17  |
| gptneox      | Windows  | MSVC 19.36.32532.0 |       |
| gptneox      | Windows  | GCC 13.1.0         |       |
| bloom        | Linux    | GCC 11.2.1         | 2.29  |
| bloom        | Windows  | MSVC 19.36.32532.0 |       |
| bloom        | Windows  | GCC 13.1.0         |       |
| starcoder    | Linux    | GCC 11.2.1         | 2.29  |
| starcoder    | Windows  | MSVC 19.36.32532.0 |       |
| starcoder    | Windows  | GCC 13.1.0         |       |



[^1]: Performance varies by use, configuration and other factors. `bigdl-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.

[^2]: *BigDL-LLM is built on top of the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [gptq](https://github.com/IST-DASLab/gptq), [ggml](https://github.com/ggerganov/ggml), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [qlora](https://github.com/artidoro/qlora), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.*
