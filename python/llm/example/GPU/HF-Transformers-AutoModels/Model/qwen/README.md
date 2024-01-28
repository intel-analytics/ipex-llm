# Qwen
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Qwen models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) as a reference Qwen model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Qwen model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install tiktoken einops transformers_stream_generator  # additional package required for Qwen-7B-Chat to conduct generation
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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Qwen model (e.g `Qwen/Qwen-7B-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
```log
Inference time: xxxx s
-------------------- Output --------------------
 AI是什么？ AI是人工智能的缩写，是指计算机科学家和工程师通过模拟人类思维和行为的方式来创造出能够自主地解决问题、学习和适应的计算机系统。
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>What is AI? <bot>
-------------------- Output --------------------
<human>What is AI? <bot>AI, or artificial intelligence, refers to the ability of a machine or computer program to perform tasks that typically require human intelligence, such as visual perception, speech recognition
```
