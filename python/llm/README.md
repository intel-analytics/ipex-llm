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
Over 20 models have been optimized/verified on `bigdl-llm`, including *LLaMA/LLaMA2, ChatGLM/ChatGLM2, MPT, Falcon, Dolly-v1/Dolly-v2, StarCoder, Whisper, InternLM, QWen, Baichuan, MOSS,* and more; see the complete list below.

<details><summary>Table of verified models</summary>
    
| Model     | Example                                                  |
|-----------|----------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/vicuna)    |
| LLaMA 2   | [link](example/transformers/transformers_int4/llama2)    |
| MPT       | [link](example/transformers/transformers_int4/mpt)       |
| Falcon    | [link](example/transformers/transformers_int4/falcon)    |
| ChatGLM   | [link](example/transformers/transformers_int4/chatglm)   | 
| ChatGLM2  | [link](example/transformers/transformers_int4/chatglm2)  | 
| Qwen      | [link](example/transformers/transformers_int4/qwen)      |
| MOSS      | [link](example/transformers/transformers_int4/moss)      | 
| Baichuan  | [link](example/transformers/transformers_int4/baichuan)  | 
| Baichuan2 | [link](example/transformers/transformers_int4/baichuan2) |
| Dolly-v1  | [link](example/transformers/transformers_int4/dolly_v1)  | 
| Dolly-v2  | [link](example/transformers/transformers_int4/dolly_v2)  | 
| RedPajama | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/redpajama) | 
| Phoenix   | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/phoenix)   | 
| StarCoder | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/starcoder) | 
| InternLM  | [link](example/transformers/transformers_int4/internlm)  |
| Whisper   | [link](example/transformers/transformers_int4/whisper)   |

</details>

### Working with `bigdl-llm`

<details><summary>Table of Contents</summary>

- [Install](#install)
- [Run Model](#run-model)
  - [Hugging Face `transformers` API](#1-hugging-face-transformers-api)
  - [Native INT4 Model](#2-native-int4-model)
  - [LangChain API](#l3-angchain-api)
  - [CLI Tool](#4-cli-tool)
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

See the complete examples [here](example/transformers/transformers_int4/).  

###### GPU INT4
You may apply INT4 optimizations to any Hugging Face *Transformers* model on Intel GPU as follows.

```python
#load Hugging Face Transformers model with INT4 optimizations
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

#run the optimized model on Intel GPU
model = model.to('xpu')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...).to('xpu')
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids.cpu())
```
See the complete examples [here](example/gpu/).

###### More Low-Bit Support
- Save and load

  After the model is optimized using `bigdl-llm`, you may save and load the model as follows:
  ```python
  model.save_low_bit(model_path)
  new_model = AutoModelForCausalLM.load_low_bit(model_path)
  ```
  *See the complete example [here](example/transformers/transformers_low_bit/).*

- Additonal data types
 
  In addition to INT4, You may apply other low bit optimizations (such as *INT8*, *INT5*, *NF4*, etc.) as follows: 

  ```python
  model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_low_bit="sym_int8")
  ```
  *See the complete example [here](example/transformers/transformers_low_bit/).*

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

See the complete example [here](example/transformers/native_int4/native_int4_pipeline.py). 

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
  See the examples [here](example/langchain/transformers_int4).
 
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

  See the examples [here](example/langchain/native_int4).

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
