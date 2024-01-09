# BlueLM
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on BlueLM models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat) as a reference BlueLM model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a BlueLM model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
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

```bash
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the BlueLM model (e.g `vivo-ai/BlueLM-7B-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'vivo-ai/BlueLM-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
[|Human|]:AI是什么？[|AI|]:
-------------------- Output --------------------
AI是什么？ AI是人工智能（Artificial Intelligence）的缩写，是一种模拟人类智能思维过程的技术。它可以让计算机系统通过学习和适应，自主地完成各种任务，
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
[|Human|]:What is AI?[|AI|]:
-------------------- Output --------------------
What is AI? AI is an AI, or artificial intelligence, that can be defined as the simulation of human intelligence processes by machines, especially computer systems.

AI is not
```
