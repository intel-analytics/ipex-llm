# DeciLM-7B
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on DeciLM-7B models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [Deci/DeciLM-7B-instruct](https://huggingface.co/Deci/DeciLM-7B-instruct) as a reference DeciLM-7B model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a DeciLM-7B model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.35.2 # required by DeciLM-7B
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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the DeciLM-7B model (e.g `Deci/DeciLM-7B-instruct`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Deci/DeciLM-7B-instruct'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [Deci/DeciLM-7B-instruct](https://huggingface.co/Deci/DeciLM-7B-instruct)

```log
Inference time: XXXX s
-------------------- Prompt --------------------
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
What is AI?
### Assistant:
-------------------- Output --------------------
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
What is AI?
### Assistant:
 AI stands for Artificial Intelligence, which refers to the development of computer systems and software that can perform tasks that typically require human intelligence, such as recognizing patterns
```
