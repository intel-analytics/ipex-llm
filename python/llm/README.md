# BigDL LLM
`bigdl-llm` is an SDK for large language model (LLM). It helps users develop AI applications that contains LLM on Intel XPU by using less computing and memory resources.`bigdl-llm` utilize a highly optimized GGML on Intel XPU.

Users could use `bigdl-llm` to
- Convert their model to lower precision
- Use command line tool like `llama.cpp` to run the model inference
- Use transformers like API to run the model inference
- Integrate the model in `langchain` pipeline

Currently `bigdl-llm` has supported
- Precision: INT4
- Model Family: llama, gptneox, bloom
- Platform: Ubuntu 20.04 or later, CentOS 7 or later, Windows 10/11
- Device: CPU
- Python: 3.9 (recommended) or later 

## Installation
BigDL-LLM is a self-contained SDK library for model loading and inferencing. Users could directly
```bash
pip install --pre --upgrade bigdl-llm
```
While model conversion procedure will rely on some 3rd party libraries. Add `[all]` option for installation to prepare environment.
```bash
pip install --pre --upgrade bigdl-llm[all]
```

## Usage
A standard procedure for using `bigdl-llm` contains 3 steps:

1. Download model from huggingface hub
2. Convert model from huggingface format to GGML format
3. Inference using `llm-cli`, transformers like API, or `langchain`.

### Convert your model
A python function and a command line tool `llm_convert` is provided to transform the model from huggingface format to GGML format.

Here is an example to use `llm_convert` command line tool.
```bash
# pth model
llm_convert -m pth -i "/path/to/llama-7b-hf/" -o "/path/to/llama-7b-int4/" -x "llama"
# gptq model
llm_convert -m gptq -i "/path/to/vicuna-13B-1.1-GPTQ-4bit-128g.pt" -o "/path/to/out.bin" -k "/path/to/tokenizer.model" -x "llama"
```

Here is an example to use `llm_convert` python API.
```bash
from bigdl.llm import llm_convert
# pth model
llm_convert(model="/path/to/llama-7b-hf/",
            outfile="/path/to/llama-7b-int4/",
            model_format="pth",
            model_family="llama")
# gptq model
llm_convert(model="/path/to/vicuna-13B-1.1-GPTQ-4bit-128g.pt",
            outfile="/path/to/out.bin",
            model_format="gptq",
            tokenizer_path="/path/to/tokenizer.model",
            model_family="llama")
```

### Inferencing

#### llm-cli command line
llm-cli is a command-line interface tool that follows the interface as the main program in `llama.cpp`.

```bash
# text completion
llm-cli -t 16 -x llama -m "/path/to/llama-7b-int4/bigdl-llm-xxx.bin" -p 'Once upon a time,'

# chatting
llm-cli -t 16 -x llama -m "/path/to/llama-7b-int4/bigdl-llm-xxx.bin" -i --color

# help information
llm-cli -x llama -h
```

#### Transformers like API
Users could load converted model or even the unconverted huggingface model directly by `AutoModelForCausalLM.from_pretrained`.

```python
from bigdl.llm.ggml.transformers import AutoModelForCausalLM

# option 1: load converted model
llm = AutoModelForCausalLM.from_pretrained("/path/to/llama-7b-int4/bigdl-llm-xxx.bin",
                                           model_family="llama")

# option 2: load huggingface checkpoint
llm = AutoModelForCausalLM.from_pretrained("/path/to/llama-7b-hf/",
                                           model_family="llama")

# option 3: load from huggingface hub repo
llm = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf",
                                           model_family="llama")
```

Users could use llm to do the inference. Apart from end-to-end fast forward, we also support split the tokenization and model inference in our API.

```python
# end-to-end fast forward w/o spliting the tokenization and model inferencing
result = llm("what is ai")

# Use transformers tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokens = tokenizer("what is ai").input_ids
tokens_id = llm.generate(tokens, max_new_tokens=32)
tokenizer.batch_decode(tokens_id)

# Use bigdl-llm tokenizer
tokens = llm.tokenize("what is ai")
tokens_id = llm.generate(tokens, max_new_tokens=32)
decoded = llm.batch_decode(tokens_id)
```

#### llama-cpp-python like API
`llama-cpp-python` has become a popular pybinding for `llama.cpp` program. Some users may be familiar with this API so `bigdl-llm` reserve this API and extend it to other model families (e.g., gptneox, bloom)

```python
from bigdl.llm.models import Llama, Bloom, Gptneox

llm = Llama("/path/to/llama-7b-int4/bigdl-llm-xxx.bin", n_threads=4)
result = llm("what is ai")
```

#### langchain integration
TODO

## Examples
We prepared several examples in https://github.com/intel-analytics/BigDL/tree/main/python/llm/example

## Dynamic library BOM
To avoid difficaulties during the installtion. `bigdl-llm` release the C implementation by dynamic library or executive file. The compilation details are stated below. **These information is only for reference, no compilation procedure is needed for our users.** `GLIBC` version may affect the compatibility.

| Model family | Platform | Compiler           | GLIBC |
| ------------ | -------- | ------------------ | ----- |
| llama        | Linux    | GCC 9.4.0          | 2.17  |
| llama        | Windows  | MSVC 19.36.32532.0 |       |
| gptneox      | Linux    | GCC 9.4.0          | 2.17  |
| gptneox      | Windows  | MSVC 19.36.32532.0 |       |
| bloom        | Linux    | GCC 9.4.0          | 2.31  |
| bloom        | Windows  | MSVC 19.36.32532.0 |       |
