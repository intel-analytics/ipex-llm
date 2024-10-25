# Run HuggingFace `transformers` Models with Pipeline Optimization on Intel NPU

In this directory, you will find examples on how to directly run HuggingFace `transformers` models with pipeline optimization on Intel NPUs. See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |

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

**Following envrionment variables are required**:

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
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder.
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `What is AI?`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-output-len MAX_OUTPUT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.

### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

```log
 Number of input tokens: 28
 Generated tokens: 32
 First token generation time: xxxx s
 Generation average latency: xxxx ms, (xxxx token/s)
 Generation time: xxxx s

Inference time: xxxx s
-------------------- Input --------------------
<s><s> [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
<s><s> [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  AI (Artificial Intelligence) is a field of computer science and technology that focuses on the development of intelligent machines that can perform
```
