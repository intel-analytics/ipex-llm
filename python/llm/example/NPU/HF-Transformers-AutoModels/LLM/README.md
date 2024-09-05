# Run HuggingFace `transformers` Models on Intel NPU
In this directory, you will find examples on how to directly run HuggingFace `transformers` models on Intel NPUs (leveraging *Intel NPU Acceleration Library*). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Chatglm3 | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) |
| Chatglm2 | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) |
| Qwen2 | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct), [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| MiniCPM | [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |
| Phi-3 | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Stablelm | [stabilityai/stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) |
| Baichuan2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan-7B-Chat) |
| Deepseek | [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) |
| Mistral | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |

## 0. Requirements
To run these examples with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
Go to https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html to download and unzip the driver.
Then go to **Device Manager**, find **Neural Processors** -> **Intel(R) AI Boost**.
Right click and select **Update Driver**. And then manually select the folder unzipped from the driver.

## 1. Install
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.10
conda activate llm

# install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
```

## 2. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
### 2.1 Configurations for Windows

> [!NOTE]
> For optimal performance, we recommend running code in `conhost` rather than Windows Terminal:
> - Press <kbd>Win</kbd>+<kbd>R</kbd> and input `conhost`, then press Enter to launch `conhost`.
> - Run following command to use conda in `conhost`. Replace `<your conda install location>` with your conda install location.
> ```
> call <your conda install location>\Scripts\activate
> ```

**Following envrionment variables are required**:

```cmd
set BIGDL_USE_NPU=1
```

## 3. Run Models
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel NPUs.

```
python ./generate.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`, and more verified models please see the list in [Verified Models](#verified-models).
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--load_in_low_bit`: argument defining the `load_in_low_bit` format used. It is default to be `sym_int8`, `sym_int4` can also be used.

### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

```log
Inference time: xxxx s
-------------------- Output --------------------
<s> Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. But her parents were always telling her to stay at home and be careful. They were worried about her safety, and they didn't want her to
--------------------------------------------------------------------------------
done
```

## 4. Run Optimized Models (Experimental)
The examples below show how to run the **_optimized HuggingFace model implementations_** on Intel NPU, including
- [Llama2-7B](./llama.py)
- [Llama3-8B](./llama.py)
- [Qwen2-1.5B](./qwen2.py)
- [Qwen2-7B](./qwen2.py)
- [MiniCPM-1B](./minicpm.py)
- [MiniCPM-2B](./minicpm.py)
- [Baichuan2-7B](./baichuan2.py)

### Recommended NPU Driver Version for MTL Users
#### 32.0.100.2540
Supported models: Llama2-7B, Llama3-8B, Qwen2-1.5B, Qwen2-7B, MiniCPM-1B, MiniCPM-2B, Baichuan2-7B

### Recommended NPU Driver Version for LNL Users
#### 32.0.100.2625
Supported models: Llama2-7B, Qwen2-1.5B, Qwen2-7B, MiniCPM-1B, Baichuan2-7B
#### 32.0.101.2715
Supported models: Llama3-8B, MiniCPM-2B

### Run
```bash
# to run Llama-2-7b-chat-hf
python llama.py

# to run Meta-Llama-3-8B-Instruct (LNL driver version: 32.0.101.2715)
python llama.py --repo-id-or-model-path meta-llama/Meta-Llama-3-8B-Instruct

# to run Qwen2-1.5B-Instruct
python qwen2.py

# to run Qwen2-7B-Instruct
python qwen2.py  --repo-id-or-model-path Qwen/Qwen2-7B-Instruct

# to run MiniCPM-1B-sft-bf16
python minicpm.py

# to run MiniCPM-2B-sft-bf16 (LNL driver version: 32.0.101.2715)
python minicpm.py --repo-id-or-model-path openbmb/MiniCPM-2B-sft-bf16

# to run Baichuan2-7B-Chat
python baichuan2.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (i.e. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--lowbit-path LOWBIT_MODEL_PATH`: argument defining the path to save/load lowbit version of the model. If it is an empty string, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded. If it is an existing path, the lowbit model in `LOWBIT_MODEL_PATH` will be loaded. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, and the converted lowbit version will be saved into `LOWBIT_MODEL_PATH`. It is default to be `''`, i.e. an empty string.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `What is AI?`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-output-len MAX_OUTPUT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: Defines the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--disable-transpose-value-cache`: Disable the optimization of transposing value cache.

### Troubleshooting

#### Output Problem
If you encounter output problem, please try to disable the optimization of transposing value cache with following command:
```bash
# to run Llama-2-7b-chat-hf
python  llama.py --disable-transpose-value-cache

# to run Meta-Llama-3-8B-Instruct (LNL driver version: 32.0.101.2715)
python llama.py --repo-id-or-model-path meta-llama/Meta-Llama-3-8B-Instruct --disable-transpose-value-cache

# to run Qwen2-1.5B-Instruct
python qwen2.py --disable-transpose-value-cache

# to run MiniCPM-1B-sft-bf16
python minicpm.py --disable-transpose-value-cache

# to run MiniCPM-2B-sft-bf16 (LNL driver version: 32.0.101.2715)
python minicpm.py --repo-id-or-model-path openbmb/MiniCPM-2B-sft-bf16 --disable-transpose-value-cache
```

#### Better Performance with High CPU Utilization
You could enable optimization by setting the environment variable with `set IPEX_LLM_CPU_LM_HEAD=1` for better performance. But this will cause high CPU utilization.


### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

```log
Inference time: xxxx s
-------------------- Input --------------------
<s><s> [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
<s><s> [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  AI (Artificial Intelligence) is a field of computer science and engineering that focuses on the development of intelligent machines that can perform tasks
```
