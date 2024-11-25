# C++ Example of running LLM on Intel NPU using IPEX-LLM (Experimental)
In this directory, you will find a C++ example on how to run LLM models on Intel NPUs using IPEX-LLM (leveraging *Intel NPU Acceleration Library*). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Qwen2 | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct), [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| Qwen2.5 | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |

## 0. Requirements
To run this C++ example with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
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

:: [optional] for Llama-3.2-1B-Instruct & Llama-3.2-3B-Instruct
pip install transformers==4.45.0 accelerate==0.33.0
```

## 2. Convert Model
We provide a [convert script](convert.py) under current directory, by running it, you can obtain the whole weights and configuration files which are required to run C++ example.

```cmd
:: to convert Qwen2.5-7b-Instruct
python convert.py --repo-id-or-model-path Qwen/Qwen2.5-7B-Instruct --save-directory <converted_model_path>

```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `Qwen/Qwen2.5-7B-Instruct`) to be downloaded, or the path to the huggingface checkpoint folder.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, and the converted model will be saved into `SAVE_DIRECTORY`.
- `--max-context-len MAX_CONTEXT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: Defines the maximum number of tokens that the input prompt can contain. It is default to be `960`.
- `--disable-transpose-value-cache`: Disable the optimization of transposing value cache.

## 3. Build C++ Example `llm-npu-cli`

You can run below cmake script in cmd to build `llm-npu-cli`, don't forget to replace below conda env dir with your own path.

```cmd
:: under current directory
:: please replace below conda env dir with your own path
set CONDA_ENV_DIR=C:\Users\arda\miniforge3\envs\llm\Lib\site-packages
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
cd Release
```

## 4. Run `llm-npu-cli`

With built `llm-npu-cli`, you can run the example with specified paramaters. For example,

```cmd
llm-npu-cli.exe -m <converted_model_path> -n 64 "AI是什么?"
```

Arguments info:
- `-m` : argument defining the path of saved converted model.
- `-n` : argument defining how many tokens will be generated.
- Last argument is your input prompt.

### 5. Sample Output
#### [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
```cmd
Input:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
AI是什么?<|im_end|>
<|im_start|>assistant


Prefill 22 tokens cost xxxx ms.
Output:
AI是"人工智能"的缩写，是英文"Artificial Intelligence"的翻译。它是研究如何使计算机也具有智能的一种技术和理论。简而言之，人工智能就是让计算机能够模仿人智能行为的一项技术。

Decode 46 tokens cost xxxx ms (avg xx.xx ms each token).
```
