# Loading GGUF models
In this directory, you will find examples on how to load GGUF model into `ipex-llm`.

## Verified Models(Q4_0)
- [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main)
- [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- [Mixtral-8x7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF)
- [Baichuan2-7B-Chat-GGUF](https://huggingface.co/second-state/Baichuan2-7B-Chat-GGUF/tree/main)
- [Bloomz-7b1-GGUF](https://huggingface.co/hzjane/bloomz-7b1-gguf)
- [falcon-7b-quantized-gguf](https://huggingface.co/xaviviro/falcon-7b-quantized-gguf/tree/main)
- [mpt-7b-chat-gguf](https://huggingface.co/maddes8cht/mosaicml-mpt-7b-chat-gguf/tree/main)

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../../README.md#system-support) for more information.

**Important: Please make sure you have installed `transformers==4.36.0` to run the example.**

## Example: Load gguf model using `from_gguf()` API
In the example [generate.py](./generate.py), we show a basic use case to load a GGUF LLaMA2 model into `ipex-llm` using `from_gguf()` API, with IPEX-LLM optimizations.

### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for IPEX-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.36.0  # upgrade transformers
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```
python ./generate.py --model <path_to_gguf_model> --prompt 'What is AI?'
```

More information about arguments can be found in [Arguments Info](#33-arguments-info) section. The expected output can be found in [Sample Output](#34-sample-output) section.

#### 3.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--model`: path to GGUF model, it should be a file with name like `llama-2-7b-chat.Q4_0.gguf`
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--low_bit`: use what low_bit to run, default is `sym_int4`.

#### 3.4 Sample Output
#### [llama-2-7b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI is a term used to describe a type of computer software that is designed to perform tasks that typically require human intelligence, such as visual perception, speech
```
