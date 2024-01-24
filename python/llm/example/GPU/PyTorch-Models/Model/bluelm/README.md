# BlueLM
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate BlueLM models. For illustration purposes, we utilize the [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat) as reference BlueLM models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a BlueLM model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
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
python ./generate.py --prompt 'AI是什么？'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the BlueLM model (e.g `vivo-ai/BlueLM-7B-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'vivo-ai/BlueLM-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat)
```log
Inference time: xxxx s
-------------------- Output --------------------
<human>AI是什么？ <bot>AI是人工智能(Artificial Intelligence)的缩写，是一种模拟人类智能思维过程的技术。它可以让计算机系统通过学习和适应，自主地进行推理、判断
```

```log
Inference time: xxxx s
-------------------- Output --------------------
<human>What is AI? <bot>AI is short for "Artificial Intelligence", which is the ability of machines to perform tasks that usually require human intelligence, such as visual perception, speech recognition,

AI is not
```