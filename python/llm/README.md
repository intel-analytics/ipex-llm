## BigDL-LLM

**`bigdl-llm`** is a library for running ***LLM*** (language language model) on your Intel ***laptop*** using INT4 with very low latency. 

*(It is built on top of the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [gptq](https://github.com/IST-DASLab/gptq), [ggml](https://github.com/ggerganov/ggml), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.)*

### Demos
See the ***optimized performance*** of `phoenix-inst-chat-7b`, `vicuna-13b-v1.1`, and `starcoder-15b` models on a 12th Gen Intel Core CPU below.

<p align="center">
            <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-7b.gif" width='33%' /> <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-13b.gif" width='33%' /> <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-15b5.gif" width='33%' />
            <img src="https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-models.png" width='85%'/>
</p>


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
2. [Hugging Face `transformer`-style API](#hugging-face-transformers-style-api)
3. [LangChain API](#langchain-api)
4. [`llama-cpp-python`-style API](#llama-cpp-python-style-api)

#### CLI Tool
Currently `bigdl-llm` CLI supports *LLaMA* (e.g., *vicuna*), *GPT-NeoX* (e.g., *redpajama*), *BLOOM* (e.g., *pheonix*) and *GPT2* (e.g., *starcoder*) model architecture; for other models, you may use the `transformer`-style or LangChain APIs.

 - ##### Convert model
 
    You may convert the downloaded model into native INT4 format using `llm-convert`.
    
   ```bash
   #convert PyTorch (fp16 or fp32) model; 
   #llama/bloom/gptneox/starcoder model family is currently supported
   lm-convert "/path/to/model/" --model-format pth --model-family "bloom" --outfile "/path/to/output/"

   #convert GPTQ-4bit model
   #only llama model family is currently supported
   llm-convert "/path/to/model/" --model-format gptq --model-family "llama" --outfile "/path/to/output/"
   ```  
  
 - ##### Run model
   
   You may run the converted model using `llm-cli` (*built on top of `main.cpp` in [llama.cpp](https://github.com/ggerganov/llama.cpp)*)

   ```bash
   #help
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -x gptneox -h

   #text completion
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -t 16 -x gptneox -m "/path/to/output/model.bin" -p 'Once upon a time,'

   #interactive mode
   #Note: The interactive mode only support LLaMA (e.g., *vicuna*), GPT-NeoX (e.g., *redpajama*) for now.
   llm-cli -m "/path/to/output/model.bin" -x llama -i

   #instruction mode with Alpaca under interactive mode
   llm-cli -m "/path/to/output/model.bin" -x llama -i --ins
   ```
   
#### Hugging Face `transformers`-style API
You may run the models using `transformers`-style API in `bigdl-llm`

- ##### Using native INT4 format

   You may convert Hugging Face *Transformers* models into native INT4 format for maximum performance as follows.

  *(Currently only llama/bloom/gptneox/starcoder model family is supported; for other models, you may use the [Hugging Face `transformers` INT4 format](#using-hugging-face-transformers-int4-format)).*

   ```python
  #convert the model
  from bigdl.llm import llm_convert
  bigdl_llm_path = llm_convert(model='/path/to/model/',
      outfile='/path/to/output/', outtype='int4', model_family="llama")

  #load the converted model
  from bigdl.llm.transformers import BigdlForCausalLM
  llm = BigdlForCausalLM.from_pretrained("/path/to/output/model.bin",...)
   
  #run the converted  model
  input_ids = llm.tokenize(prompt)
  output_ids = llm.generate(input_ids, ...)
  output = llm.batch_decode(output_ids)
  ``` 

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

#### LangChain API
You may convert Hugging Face *Transformers* models into *native INT4* format (currently only *llama*/*bloom*/*gptneox*/*starcoder* model family is supported), and then run the converted models using the LangChain API in `bigdl-llm` as follows.

```python
from bigdl.llm.langchain.llms import BigdlLLM
from bigdl.llm.langchain.embeddings import BigdlLLMEmbeddings
from langchain.chains.question_answering import load_qa_chain

embeddings = BigdlLLMEmbeddings(model_path='/path/to/converted/model.bin',
                                model_family="llama",...)
bigdl_llm = BigdlLLM(model_path='/path/to/converted/model.bin',
                     model_family="llama",...)

doc_chain = load_qa_chain(bigdl_llm, ...)
doc_chain.run(...)
```

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
| llama        | Linux    | GCC 9.3.1          | 2.17  |
| llama        | Windows  | MSVC 19.36.32532.0 |       |
| gptneox      | Linux    | GCC 9.3.1          | 2.17  |
| gptneox      | Windows  | MSVC 19.36.32532.0 |       |
| bloom        | Linux    | GCC 9.4.0          | 2.29  |
| bloom        | Windows  | MSVC 19.36.32532.0 |       |
| starcoder    | Linux    | GCC 9.4.0          | 2.29  |
| starcoder    | Windows  | MSVC 19.36.32532.0 |       |
