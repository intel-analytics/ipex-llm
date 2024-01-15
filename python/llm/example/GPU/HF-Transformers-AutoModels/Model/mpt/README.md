# MPT
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Llama2 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat) as a reference MPT model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an MPT model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install einops  # additional package required for mpt-7b-chat and mpt-30b-chat to conduct generation
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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the MPT model (e.g. `mosaicml/mpt-7b-chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'mosaicml/mpt-7b-chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant

-------------------- Output --------------------
user
What is AI?
assistant
AI, or artificial intelligence, is the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require
```
