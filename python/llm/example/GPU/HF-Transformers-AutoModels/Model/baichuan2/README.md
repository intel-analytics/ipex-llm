# Baichuan
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Baichuan2 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan-7B-Chat) as a reference Baichuan model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Baichuan model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers_stream_generator  # additional package required for Baichuan-7B-Chat to conduct generation
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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Baichuan model (e.g `baichuan-inc/Baichuan2-7B-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'baichuan-inc/Baichuan2-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
```log
-------------------- Prompt --------------------
<human>AI是什么？ <bot>
-------------------- Output --------------------
<human>AI是什么？ <bot>
AI是人工智能（Artificial Intelligence）的缩写，它是指让计算机或机器模拟、扩展和辅助人类的智能。AI技术已经广泛应用于各个领域
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>What is AI? <bot>
-------------------- Output --------------------
<human>What is AI? <bot>Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that would typically require human intelligence. These tasks include learning, reasoning, problem
```
