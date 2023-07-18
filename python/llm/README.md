## BigDL-LLM

**`bigdl-llm`** is a library for running ***LLM*** (language language model) on your Intel ***laptop*** using INT4 with very low latency (for any Hugging Face *Transformers* model). 

*(It is built on top of the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [gptq](https://github.com/IST-DASLab/gptq), [ggml](https://github.com/ggerganov/ggml), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.)*

### Demos
See the ***optimized performance*** of `phoenix-inst-chat-7b`, `vicuna-13b-v1.1`, and `starcoder-15b` models on a 12th Gen Intel Core CPU below.

<p align="center">
            <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-7b.gif" width='33%' /> <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-13b.gif" width='33%' /> <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-15b5.gif" width='33%' />
            <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-models.png" width='85%'/>
</p>

### Verified models
We may use any Hugging Face Transfomer models on `bigdl-llm`, and the following models have been verified on Intel laptops.
| Model     | Example                                                  |
|-----------|----------------------------------------------------------|
| LLaMA     | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/vicuna)    |
| MPT       | [link](example/transformers/transformers_int4/mpt)       |
| Falcon    | [link](example/transformers/transformers_int4/falcon)    |
| ChatGLM   | [link](example/transformers/transformers_int4/chatglm)   | 
| ChatGLM2  | [link](example/transformers/transformers_int4/chatglm2)  | 
| MOSS      | [link](example/transformers/transformers_int4/moss)      | 
| Baichuan  | [link](example/transformers/transformers_int4/baichuan)  | 
| Dolly-v1  | [link](example/transformers/transformers_int4/dolly_v1)  | 
| RedPajama | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/redpajama) | 
| Phoenix   | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/phoenix)   | 
| StarCoder | [link1](example/transformers/native_int4), [link2](example/transformers/transformers_int4/starcoder) | 


### Working with `bigdl-llm`

<details><summary>Table of Contents</summary>

- [Install](#install)
- [Download Model](#download-model)
- [Run Model](#run-model)
  - [CLI Tool](#cli-tool)
  - [Hugging Face `transformers`-style API](#hugging-face-transformers-style-api)
  - [LangChain API](#langchain-api)
  - [`llama-cpp-python`-style API](#llama-cpp-python-style-api)
- [`bigdl-llm` Dependence](#bigdl-llm-dependence)

</details>

#### Install
You may install **`bigdl-llm`** as follows:
```bash
pip install --pre --upgrade bigdl-llm[all]
```
#### Download Model

You may download any PyTorch model in Hugging Face *Transformers* format (including *FP16* or *FP32* or *GPTQ-4bit*).

#### Run Model
 
You may run the models using **`bigdl-llm`** through one of the following APIs:
1. [CLI (command line interface) Tool](#cli-tool)
2. [Hugging Face `transformers`-style API](#hugging-face-transformers-style-api)
3. [LangChain API](#langchain-api)
4. [`llama-cpp-python`-style API](#llama-cpp-python-style-api)

#### CLI Tool
>**Note**: Currently `bigdl-llm` CLI supports *LLaMA* (e.g., *vicuna*), *GPT-NeoX* (e.g., *redpajama*), *BLOOM* (e.g., *pheonix*) and *GPT2* (e.g., *starcoder*) model architecture; for other models, you may use the `transformers`-style or LangChain APIs.

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

#### Hugging Face `transformers`-style API
You may run the models using `transformers`-style API in `bigdl-llm`.

- ##### Using Hugging Face `transformers` INT4 format

  You may apply INT4 optimizations to any Hugging Face *Transformers* models as follows.

  ```python
  #load Hugging Face Transformers model with INT4 optimizations
  from bigdl.llm.transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)

  #run the optimized model
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  input_ids = tokenizer.encode(input_str, ...)
  output_ids = model.generate(input_ids, ...)
  output = tokenizer.batch_decode(output_ids)
  ```

  See the complete examples [here](example/transformers/transformers_int4/).  

  >**Note**: You may apply more low bit optimizations (including INT8, INT5 and INT4) as follows: 
  >```python
  >model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_low_bit="sym_int5")
  >```
  >See the complete example [here](example/transformers/transformers_low_bit/).

  
  After the model is optimizaed using INT4 (or INT8/INT5), you may save and load the optimized model as follows:
  ```python
  model.save_low_bit(model_path)
  
  new_model = AutoModelForCausalLM.load_low_bit(model_path)
  ```
  See the example [here](example/transformers/transformers_low_bit/).

- ##### Using native INT4 format

  You may also convert Hugging Face *Transformers* models into native INT4 format for maximum performance as follows.

  >**Note**: Currently only llama/bloom/gptneox/starcoder model family is supported; for other models, you may use the Transformers INT4 format as described above).

   ```python
  #convert the model
  from bigdl.llm import llm_convert
  bigdl_llm_path = llm_convert(model='/path/to/model/',
          outfile='/path/to/output/', outtype='int4', model_family="llama")

  #load the converted model
  from bigdl.llm.transformers import BigdlNativeForCausalLM
  llm = BigdlNativeForCausalLM.from_pretrained("/path/to/output/model.bin",...)
   
  #run the converted  model
  input_ids = llm.tokenize(prompt)
  output_ids = llm.generate(input_ids, ...)
  output = llm.batch_decode(output_ids)
  ``` 

  See the complete example [here](example/transformers/native_int4/native_int4_pipeline.py). 

#### LangChain API
You may run the models using the LangChain API in `bigdl-llm`.

- **Using Hugging Face `transformers` INT4 format**

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
 
- **Using native INT4 format**

  You may also convert Hugging Face *Transformers* models into *native INT4* format (currently only *llama*/*bloom*/*gptneox*/*starcoder* model family is supported), and then run the converted models using the LangChain API as follows.
  
  >**Note**: Currently only llama/bloom/gptneox/starcoder model family is supported; for other models, you may use the Transformers INT4 format as described above).

  ```python
  from bigdl.llm.langchain.llms import BigdlNativeLLM
  from bigdl.llm.langchain.embeddings import BigdlNativeEmbeddings
  from langchain.chains.question_answering import load_qa_chain

  embeddings = BigdlNativeEmbeddings(model_path='/path/to/converted/model.bin',
                            model_family="llama",...)
  bigdl_llm = BigdlNativeLLM(model_path='/path/to/converted/model.bin',
                       model_family="llama",...)

  doc_chain = load_qa_chain(bigdl_llm, ...)
  doc_chain.run(...)
  ```

  See the examples [here](example/langchain/native_int4).

#### `llama-cpp-python`-style API

You may also run the converted models using the `llama-cpp-python`-style API in `bigdl-llm` as follows.

```python
from bigdl.llm.models import Llama, Bloom, Gptneox

llm = Bloom("/path/to/converted/model.bin", n_threads=4)
result = llm("what is ai")
```

### `bigdl-llm` Dependence 
The native code/lib in `bigdl-llm` has been built using the following tools; in particular, lower  `LIBC` version on your Linux system may be incompatible with `bigdl-llm`.

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
