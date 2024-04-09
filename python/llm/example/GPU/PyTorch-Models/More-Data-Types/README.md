# IPEX-LLM Low Bit Optimization for Large Language Model

In this example, we show how to apply IPEX-LLM low-bit optimizations (including INT8/INT5/INT4) to Llama2 model, and then run inference on the optimized low-bit model with Intel GPUs.

## 0. Requirements
To run this example with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../README.md#system-support) for more information.

## Example: Load Model in Low-Bit Optimization
In the example [generate.py](./generate.py), we show a basic use case of low-bit optimizations (including INT8/INT5/INT4) on a Llama2 model to predict the next N tokens using `generate()` API. By specifying `--low-bit` argument, you could apply other low-bit optimization (e.g. INT8/INT5) on model.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
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

Following command will load and run model in symmetric int 8 optimization:
```
python ./generate.py --low-bit sym_int8
```
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--low-bit`: argument defining the low-bit optimization data type, options are sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8. (sym_int4 means symmetric int 4, asym_int4 means asymmetric int 4, etc.). Relevant low bit optimizations will be applied to the model. It is default to be `sym_int8`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

### 4. Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech
```

#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI, or Artificial Intelligence, refers to the ability of machines to perform tasks that would normally require human intelligence, such as learning, problem-
```