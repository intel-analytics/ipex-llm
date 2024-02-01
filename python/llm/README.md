## BigDL-LLM
**[`bigdl-llm`](https://bigdl.readthedocs.io/en/latest/doc/LLM/index.html)** is a library for running **LLM** (large language model) on Intel **XPU** (from *Laptop* to *GPU* to *Cloud*) using **INT4** with very low latency[^1] (for any **PyTorch** model).

> *It is built on top of the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [gptq](https://github.com/IST-DASLab/gptq), [ggml](https://github.com/ggerganov/ggml), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [qlora](https://github.com/artidoro/qlora), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.*
    
### Demos
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

### Verified models
Over 20 models have been optimized/verified on `bigdl-llm`, including *LLaMA/LLaMA2, ChatGLM/ChatGLM2, Mistral, Falcon, MPT, Dolly, StarCoder, Whisper, Baichuan, InternLM, QWen, Aquila, MOSS,* and more; see the complete list below.
  
| Model      | CPU Example                                                    | GPU Example                                                     |
|------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](example/GPU/HF-Transformers-AutoModels/Model/vicuna)|
| LLaMA 2    | [link1](example/CPU/Native-Models), [link2](example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link1](example/GPU/HF-Transformers-AutoModels/Model/llama2), [link2-low GPU memory example](example/GPU/PyTorch-Models/Model/llama2#example-2---low-memory-version-predict-tokens-using-generate-api) |
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
| BlueLM | [link](example/CPU/HF-Transformers-AutoModels/Model/bluelm) | [link](example/GPU/HF-Transformers-AutoModels/Model/bluelm) |
| SOLAR | [link](example/CPU/HF-Transformers-AutoModels/Model/solar) | [link](example/GPU/HF-Transformers-AutoModels/Model/solar) |

### Working with `bigdl-llm`

<details><summary>Table of Contents</summary>

- [BigDL-LLM](#bigdl-llm)
  - [Demos](#demos)
  - [Verified models](#verified-models)
  - [Working with `bigdl-llm`](#working-with-bigdl-llm)
    - [Install](#install)
      - [CPU](#cpu)
      - [GPU](#gpu)
    - [Run Model](#run-model)
      - [1. Hugging Face `transformers` API](#1-hugging-face-transformers-api)
        - [CPU INT4](#cpu-int4)
        - [GPU INT4](#gpu-int4)
        - [More Low-Bit Support](#more-low-bit-support)
      - [2. Native INT4 model](#2-native-int4-model)
      - [3. LangChain API](#3-langchain-api)
      - [4. CLI Tool](#4-cli-tool)
  - [`bigdl-llm` API Doc](#bigdl-llm-api-doc)
  - [`bigdl-llm` Dependency](#bigdl-llm-dependency)

</details>

#### Install
##### CPU
You may install **`bigdl-llm`** on Intel CPU as follows:
```bash
pip install --pre --upgrade bigdl-llm[all]
```
> Note: `bigdl-llm` has been tested on Python 3.9

##### GPU
You may install **`bigdl-llm`** on Intel GPU as follows:
```bash
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
> Note: `bigdl-llm` has been tested on Python 3.9

#### Run Model
 
You may run the models using **`bigdl-llm`** through one of the following APIs:
1. [Hugging Face `transformers` API](#1-hugging-face-transformers-api)
2. [Native INT4 Model](#2-native-int4-model)
3. [LangChain API](#3-langchain-api)
4. [CLI (command line interface) Tool](#4-cli-tool)

##### 1. Hugging Face `transformers` API
You may run any Hugging Face *Transformers* model as follows:

###### CPU INT4
You may apply INT4 optimizations to any Hugging Face *Transformers* model on Intel CPU as follows.

```python
#load Hugging Face Transformers model with INT4 optimizations
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

#run the optimized model on Intel CPU
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...)
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids)
```

See the complete examples [here](example/CPU/HF-Transformers-AutoModels/Model/).  

###### GPU INT4
You may apply INT4 optimizations to any Hugging Face *Transformers* model on Intel GPU as follows.

```python
#load Hugging Face Transformers model with INT4 optimizations
from bigdl.llm.transformers import AutoModelForCausalLM
import intel_extension_for_pytorch
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

#run the optimized model on Intel GPU
model = model.to('xpu')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...).to('xpu')
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids.cpu())
```
See the complete examples [here](example/GPU).

###### More Low-Bit Support
- Save and load

  After the model is optimized using `bigdl-llm`, you may save and load the model as follows:
  ```python
  model.save_low_bit(model_path)
  new_model = AutoModelForCausalLM.load_low_bit(model_path)
  ```
  *See the complete example [here](example/CPU/HF-Transformers-AutoModels/Save-Load).*

- Additonal data types
 
  In addition to INT4, You may apply other low bit optimizations (such as *INT8*, *INT5*, *NF4*, etc.) as follows: 

  ```python
  model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_low_bit="sym_int8")
  ```
  *See the complete example [here](example/CPU/HF-Transformers-AutoModels/More-Data-Types).*

##### 2. Native INT4 model
 
You may also convert Hugging Face *Transformers* models into native INT4 model format for maximum performance as follows.

>**Notes**: Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; for other models, you may use the Hugging Face `transformers` model format as described above).
  
```python
#convert the model
from bigdl.llm import llm_convert
bigdl_llm_path = llm_convert(model='/path/to/model/',
        outfile='/path/to/output/', outtype='int4', model_family="llama")

#load the converted model
#switch to ChatGLMForCausalLM/GptneoxForCausalLM/BloomForCausalLM/StarcoderForCausalLM to load other models
from bigdl.llm.transformers import LlamaForCausalLM
llm = LlamaForCausalLM.from_pretrained("/path/to/output/model.bin", native=True, ...)
  
#run the converted model
input_ids = llm.tokenize(prompt)
output_ids = llm.generate(input_ids, ...)
output = llm.batch_decode(output_ids)
``` 

See the complete example [here](example/CPU/Native-Models/native_int4_pipeline.py). 

##### 3. LangChain API
You may run the models using the LangChain API in `bigdl-llm`.

- **Using Hugging Face `transformers` model**

  You may run any Hugging Face *Transformers* model (with INT4 optimiztions applied) using the LangChain API as follows:

  ```python
  from bigdl.llm.langchain.llms import TransformersLLM
  from bigdl.llm.langchain.embeddings import TransformersEmbeddings
  from langchain.chains.question_answering import load_qa_chain

  embeddings = TransformersEmbeddings.from_model_id(model_id=model_path)
  bigdl_llm = TransformersLLM.from_model_id(model_id=model_path, ...)

  doc_chain = load_qa_chain(bigdl_llm, ...)
  output = doc_chain.run(...)
  ```
  See the examples [here](example/CPU/LangChain/transformers_int4).
 
- **Using native INT4 model**

  You may also convert Hugging Face *Transformers* models into *native INT4* format, and then run the converted models using the LangChain API as follows.
  
  >**Notes**:* Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; for other models, you may use the Hugging Face `transformers` model format as described above).

  ```python
  from bigdl.llm.langchain.llms import LlamaLLM
  from bigdl.llm.langchain.embeddings import LlamaEmbeddings
  from langchain.chains.question_answering import load_qa_chain

  #switch to ChatGLMEmbeddings/GptneoxEmbeddings/BloomEmbeddings/StarcoderEmbeddings to load other models
  embeddings = LlamaEmbeddings(model_path='/path/to/converted/model.bin')
  #switch to ChatGLMLLM/GptneoxLLM/BloomLLM/StarcoderLLM to load other models
  bigdl_llm = LlamaLLM(model_path='/path/to/converted/model.bin')

  doc_chain = load_qa_chain(bigdl_llm, ...)
  doc_chain.run(...)
  ```

  See the examples [here](example/CPU/LangChain/native_int4).

##### 4. CLI Tool
>**Note**: Currently `bigdl-llm` CLI supports *LLaMA* (e.g., *vicuna*), *GPT-NeoX* (e.g., *redpajama*), *BLOOM* (e.g., *pheonix*) and *GPT2* (e.g., *starcoder*) model architecture; for other models, you may use the Hugging Face `transformers` or LangChain APIs.

 - ##### Convert model
 
    You may convert the downloaded model into native INT4 format using `llm-convert`.
    
   ```bash
   #convert PyTorch (fp16 or fp32) model; 
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-convert "/path/to/model/" --model-format pth --model-family "bloom" --outfile "/path/to/output/"

   #convert GPTQ-4bit model
   #only llama model family is currently supported
   llm-convert "/path/to/model/" --model-format gptq --model-family "llama" --outfile "/path/to/output/"
   ```  
  
 - ##### Run model
   
   You may run the converted model using `llm-cli` or `llm-chat` (*built on top of `main.cpp` in [llama.cpp](https://github.com/ggerganov/llama.cpp)*)

   ```bash
   #help
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -x gptneox -h

   #text completion
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -t 16 -x gptneox -m "/path/to/output/model.bin" -p 'Once upon a time,'

   #chat mode
   #llama/gptneox model family is currently supported
   llm-chat -m "/path/to/output/model.bin" -x llama
   ```

### `bigdl-llm` API Doc
See the inital `bigdl-llm` API Doc [here](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/LLM/index.html).

[^1]: Performance varies by use, configuration and other factors. `bigdl-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.

### `bigdl-llm` Dependency
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
