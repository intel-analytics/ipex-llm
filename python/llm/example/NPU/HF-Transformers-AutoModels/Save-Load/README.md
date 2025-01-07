# Save/Load Low-Bit Models with IPEX-LLM Optimizations

In this directory, you will find example on how you could save/load models with IPEX-LLM optimizations on Intel NPU.

## Example: Save/Load Optimized Models
In the example [generate.py](./generate.py), we show a basic use case of saving/loading model in low-bit optimizations to predict the next N tokens using `generate()` API.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

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
```
Please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 3. Running examples

If you want to save the optimized model, run:
```
python ./generate.py --repo-id-or-model-path "meta-llama/Llama-2-7b-chat-hf" --save-directory path/to/save/model
```

If you want to load the optimized low-bit model, run:
```
python ./generate.py --load-directory path/to/load/model
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model to be downloaded, or the path to the ModelScope checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--save-directory`: argument defining the path to save the low-bit model. Then you can load the low-bit directly.
- `--load-directory`: argument defining the path to load low-bit model.
- `--prompt PROMPT`: argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-context-len MAX_CONTEXT_LEN`: argument defining the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: argument defining the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model. Current available options are `"sym_int4"`, `"asym_int4"` and `"sym_int8"`, with `"sym_int4"` as the default.

### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Input --------------------
<s><s>  [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]

-------------------- Output --------------------
<s><s>  [INST] <<SYS>>

<</SYS>>

What is AI? [/INST]

Artificial Intelligence (AI) is a field of computer science and technology that focuses on the development of intelligent machines that can perform tasks that
```
