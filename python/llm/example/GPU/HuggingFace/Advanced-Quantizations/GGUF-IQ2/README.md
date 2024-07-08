# GGUF-IQ2

This example shows how to run INT2 models using the IQ2 mechanism (first implemented by `llama.cpp`) in IPEX-LLM on Intel GPU.

## Verified Models

- [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), using [llama-v2-7b.imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/resolve/main/llama-v2-7b.imatrix)
- [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), using [llama-v2-7b.imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/resolve/main/llama-v2-7b.imatrix)
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), using [mistral-7b-instruct-v0.2.imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/resolve/main/mistral-7b-instruct-v0.2.imatrix)
- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), using [mixtral-8x7b.imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/resolve/main/mixtral-8x7b.imatrix)
- [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), using [mixtral-8x7b-instruct-v0.1.imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/resolve/main/mixtral-8x7b-instruct-v0.1.imatrix)

## Requirements

To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a GGUF-IQ2 model to predict the next N tokens using `generate()` API, with IPEX-LLM optimizations.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.35.0
```
**Note: For Mixtral model, please use transformers 4.36.0:**
```bash
pip install transformers==4.36.0
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
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output

#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
### HUMAN:
What is AI?

### RESPONSE:

-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

Artificial intelligence (AI) refers to the ability of machines to perform tasks that would typically require human intelligence, such as learning, problem-solving
```
