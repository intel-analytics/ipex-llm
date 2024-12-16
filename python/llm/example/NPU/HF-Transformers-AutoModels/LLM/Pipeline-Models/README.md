# Run HuggingFace `transformers` Models with Pipeline Optimization on Intel NPU

In this directory, you will find examples on how to directly run HuggingFace `transformers` models with pipeline optimization on Intel NPUs. See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Llama3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Qwen2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| Qwen2.5 | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Baichuan2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan-7B-Chat) |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quick Start](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install & Runtime Configurations
### 1.1 Installation on Windows
Please refer to [Quick Start](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-ipex-llm-with-npu-support) for `ipex-llm` installation.

### 1.2 Runtime Configurations
Please refer to [Quick Start](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Run Models
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel NPUs.

```cmd
:: to run Llama-2-7b-chat-hf
python llama2.py --repo-id-or-model-path "meta-llama/Llama-2-7b-chat-hf" --save-directory <converted_model_path>

:: to run Meta-Llama-3-8B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --save-directory <converted_model_path>

:: to run Llama-3.2-1B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Llama-3.2-1B-Instruct" --save-directory <converted_model_path>

:: to run Llama-3.2-3B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Llama-3.2-3B-Instruct" --save-directory <converted_model_path>

:: to run Qwen2.5-7B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2.5-7B-Instruct" --save-directory <converted_model_path>

:: to run Qwen2-1.5B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2-1.5B-Instruct" --low-bit sym_int8 --save-directory <converted_model_path>

:: to run Qwen2.5-3B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2.5-3B-Instruct" --low-bit sym_int8 --save-directory <converted_model_path>

:: to run Baichuan2-7B-Chat
python baichuan2.py --repo-id-or-model-path "baichuan-inc/Baichuan2-7B-Chat" --save-directory <converted_model_path>

:: to run MiniCPM-1B-sft-bf16
python minicpm.py --repo-id-or-model-path "openbmb/MiniCPM-1B-sft-bf16" --save-directory <converted_model_path>

:: to run MiniCPM-2B-sft-bf16
python minicpm.py --repo-id-or-model-path "openbmb/MiniCPM-2B-sft-bf16" --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder.
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `What is AI?`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-context-len MAX_CONTEXT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: Defines the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--disable-transpose-value-cache`: Disable the optimization of transposing value cache.
- `--disable-streaming`: Disable streaming mode of generation.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

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
