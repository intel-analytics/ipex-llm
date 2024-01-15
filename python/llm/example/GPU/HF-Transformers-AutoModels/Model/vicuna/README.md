# Vicuna
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Vicuna models. For illustration purposes, we utilize the [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) and [eachadea/vicuna-7b-1.1](https://huggingface.co/eachadea/vicuna-7b-1.1) as reference Vicuna models.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Vicuna model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.


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

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Vicuna model (e.g. `lmsys/vicuna-13b-v1.3` and `eachadea/vicuna-7b-1.1`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'lmsys/vicuna-13b-v1.3'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Vicuna model based on the capabilities of your machine.

#### Sample Output
#### [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
### Human:
What is AI? 
 ### Assistant:

-------------------- Output --------------------
### Human:
What is AI? 
 ### Assistant:
AI, or Artificial Intelligence, refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception,
```

#### [eachadea/vicuna-7b-1.1](https://huggingface.co/eachadea/vicuna-7b-1.1)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
### Human:
What is AI? 
 ### Assistant:

-------------------- Output --------------------
### Human:
What is AI? 
 ### Assistant:
AI, or artificial intelligence, refers to the ability of a machine or computer program to mimic human intelligence and perform tasks that would normally require human intelligence to
```
