# Run HuggingFace `transformers` Models on Intel NPU
In this directory, you will find examples on how to directly run HuggingFace `transformers` models on Intel NPUs (leveraging *Intel NPU Acceleration Library*). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Llama3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| GLM-Edge | [THUDM/glm-edge-1.5b-chat](https://huggingface.co/THUDM/glm-edge-1.5b-chat), [THUDM/glm-edge-4b-chat](https://huggingface.co/THUDM/glm-edge-4b-chat) |
| Qwen2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |
| Qwen2.5 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |
| Baichuan2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |

Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#python-api) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install & Runtime Configurations
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```cmd
conda create -n llm python=3.11
conda activate llm

:: install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]

:: [optional] for Llama-3.2-1B-Instruct & Llama-3.2-3B-Instruct
pip install transformers==4.45.0 accelerate==0.33.0

:: [optional] for glm-edge-1.5b-chat & glm-edge-4b-chat
pip install transformers==4.47.0 accelerate==0.26.0
```

Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-ipex-llm-with-npu-support) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Run Optimized Models
The examples below show how to run the **_optimized HuggingFace model implementations_** on Intel NPU, including
- [Llama2-7B](./llama2.py)
- [Llama3-8B](./llama3.py)
- [Llama3.2-1B](./llama3.py)
- [Llama3.2-3B](./llama3.py)
- [GLM-Edge-1.5B](./glm.py)
- [GLM-Edge-4B](./glm.py)
- [Qwen2-1.5B](./qwen.py)
- [Qwen2-7B](./qwen.py)
- [Qwen2.5-3B](./qwen.py)
- [Qwen2.5-7B](./qwen.py)
- [DeepSeek-R1-Distill-Qwen-1.5B](./qwen.py)
- [DeepSeek-R1-Distill-Qwen-7B](./qwen.py)
- [MiniCPM-1B](./minicpm.py)
- [MiniCPM-2B](./minicpm.py)
- [Baichuan2-7B](./baichuan2.py)

### Run
```cmd
:: to run Llama-2-7b-chat-hf
python llama2.py --repo-id-or-model-path "meta-llama/Llama-2-7b-chat-hf" --save-directory <converted_model_path>

:: to run Meta-Llama-3-8B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --save-directory <converted_model_path>

:: to run Llama-3.2-1B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Llama-3.2-1B-Instruct" --save-directory <converted_model_path>

:: to run Llama-3.2-3B-Instruct
python llama3.py --repo-id-or-model-path "meta-llama/Llama-3.2-3B-Instruct" --save-directory <converted_model_path>

:: to run glm-edge-1.5b-chat
python glm.py --repo-id-or-model-path "THUDM/glm-edge-1.5b-chat" --save-directory <converted_model_path>

:: to run glm-edge-4b-chat
python glm.py --repo-id-or-model-path "THUDM/glm-edge-4b-chat" --save-directory <converted_model_path>

:: to run Qwen2-1.5B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2-1.5B-Instruct"  --save-directory <converted_model_path>

:: to run Qwen2-7B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2-7B-Instruct" --save-directory <converted_model_path>

:: to run Qwen2.5-3B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2.5-3B-Instruct" --low-bit asym_int4 --save-directory <converted_model_path>

:: to run Qwen2.5-7B-Instruct
python qwen.py --repo-id-or-model-path "Qwen/Qwen2.5-7B-Instruct" --save-directory <converted_model_path>

:: to run DeepSeek-R1-Distill-Qwen-1.5B
python qwen.py --repo-id-or-model-path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --save-directory <converted_model_path>

:: to run DeepSeek-R1-Distill-Qwen-7B
python qwen.py --repo-id-or-model-path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --save-directory <converted_model_path>

:: to run MiniCPM-1B-sft-bf16
python minicpm.py --repo-id-or-model-path "openbmb/MiniCPM-1B-sft-bf16" --save-directory <converted_model_path>

:: to run MiniCPM-2B-sft-bf16
python minicpm.py --repo-id-or-model-path "openbmb/MiniCPM-2B-sft-bf16" --save-directory <converted_model_path>

:: to run Baichuan2-7B-Chat
python baichuan2.py --repo-id-or-model-path "baichuan-inc/Baichuan2-7B-Chat" --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g.`Meta-llama/Llama-2-7b-chat-hf` for Llama2-7B) to be downloaded, or the path to the huggingface checkpoint folder.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `"What is AI?"` or `"AI是什么？"`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-context-len MAX_CONTEXT_LEN`: argument defining the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: argument defining the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model. Current available options are `"sym_int4"`, `"asym_int4"` and `"sym_int8"`, with `"sym_int4"` as the default.
- `--disable-streaming`: argument defining whether to disable the streaming mode for generation.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

### Troubleshooting

#### Accuracy Tuning
If you enconter output issues when running the examples, you could try the following methods to tune the accuracy:

1. Before running the example, consider setting an additional environment variable `IPEX_LLM_NPU_QUANTIZATION_OPT=1` to enhance output quality.

2. If you are using the default `LOW_BIT` value (i.e. `sym_int4` optimizations), you could try to use `--low-bit "asym_int4"` instead to tune the output quality.

3. You could refer to the [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#accuracy-tuning) for more accuracy tuning strategies.

> [!IMPORTANT]
> Please note that to make the above methods taking effect, you must specify a new folder for `SAVE_DIRECTORY`. Reusing the same `SAVE_DIRECTORY` will load the previously saved low-bit model, and thus making the above accuracy tuning strategies ineffective.

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
