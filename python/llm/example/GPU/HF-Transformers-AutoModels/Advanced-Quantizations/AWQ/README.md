# AWQ

This example shows how to directly run 4-bit AWQ models using IPEX-LLM on Intel GPU.

## Verified Models

### Auto-AWQ Backend
- [Llama-2-7B-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ)
- [CodeLlama-7B-AWQ](https://huggingface.co/TheBloke/CodeLlama-7B-AWQ)
- [Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ)
- [Mistral-7B-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-v0.1-AWQ)
- [vicuna-7B-v1.5-AWQ](https://huggingface.co/TheBloke/vicuna-7B-v1.5-AWQ)
- [vicuna-13B-v1.5-AWQ](https://huggingface.co/TheBloke/vicuna-13B-v1.5-AWQ)
- [llava-v1.5-13B-AWQ](https://huggingface.co/TheBloke/llava-v1.5-13B-AWQ)
- [Yi-6B-AWQ](https://huggingface.co/TheBloke/Yi-6B-AWQ)
- [Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)
- [Mixtral-8x7B-Instruct-v0.1-AWQ](https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-AWQ)

### llm-AWQ Backend

- [vicuna-7b-1.5-awq](https://huggingface.co/ybelkada/vicuna-7b-1.5-awq)

## Requirements

To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a AWQ model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.35.0
pip install autoawq==0.1.8 --no-deps
pip install accelerate==0.25.0
pip install einops
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the AWQ model (e.g. `TheBloke/Llama-2-7B-Chat-AWQ`, `TheBloke/Mistral-7B-Instruct-v0.1-AWQ`, `TheBloke/Mistral-7B-v0.1-AWQ`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'TheBloke/Llama-2-7B-Chat-AWQ'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Llama2 model based on the capabilities of your machine.

#### 2.3 Sample Output

#### [&#34;TheBloke/Llama-2-7B-Chat-AWQ&#34;](https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ)

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

Artificial intelligence (AI) is the ability of machines to perform tasks that typically require human intelligence, such as learning, problem-solving, decision
```
