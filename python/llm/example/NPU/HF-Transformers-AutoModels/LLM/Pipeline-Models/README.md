# Run HuggingFace `transformers` Models with Pipeline Optimization on Intel NPU

In this directory, you will find examples on how to directly run HuggingFace `transformers` models with pipeline optimization on Intel NPUs. See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Qwen2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| Qwen2.5 | [Qwen/Qwen2.5-7b-Instruct](https://huggingface.co/Qwen/Qwen2.5-7b-Instruct) |
| Baichuan2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan-7B-Chat) |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |

## 0. Requirements
To run these examples with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
Go to https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html to download and unzip the driver.
Then go to **Device Manager**, find **Neural Processors** -> **Intel(R) AI Boost**.
Right click and select **Update Driver** -> **Browse my computer for drivers**. And then manually select the unzipped driver folder to install.

## 1. Install
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```cmd
conda create -n llm python=3.10
conda activate llm

:: install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
```

## 2. Runtime Configurations

**Following environment variables are required**:

```cmd
set BIGDL_USE_NPU=1
```

## 3. Run Models
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel NPUs.

```cmd
:: to run Llama-2-7b-chat-hf
python llama2.py

:: to run Meta-Llama-3-8B-Instruct
python llama3.py

:: to run Qwen2.5-7b-Instruct
python qwen.py

:: to run Qwen2-1.5b-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2-1.5B-Instruct" --load_in_low_bit "sym_int8"

:: to run Baichuan2-7B-Chat
python baichuan2.py

:: to run MiniCPM-1B-sft-bf16
python minicpm.py

:: to run MiniCPM-2B-sft-bf16
python minicpm.py --repo-id-or-model-path "openbmb/MiniCPM-2B-sft-bf16"
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder.
- `--lowbit-path LOWBIT_MODEL_PATH`: argument defining the path to save/load lowbit version of the model. If it is an empty string, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded. If it is an existing path, the lowbit model in `LOWBIT_MODEL_PATH` will be loaded. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, and the converted lowbit version will be saved into `LOWBIT_MODEL_PATH`. It is default to be `''`, i.e. an empty string.
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `What is AI?`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-context-len MAX_CONTEXT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: Defines the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--disable-transpose-value-cache`: Disable the optimization of transposing value cache.
- `--disable-streaming`: Disable streaming mode of generation.

### Sample Output of Streaming Mode
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
 
```log
-------------------- Input --------------------
input length: 28
<s>[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
 AI (Artificial Intelligence) is a field of computer science and technology that focuses on the development of intelligent machines that can perform

Inference time: xxxx s
```
