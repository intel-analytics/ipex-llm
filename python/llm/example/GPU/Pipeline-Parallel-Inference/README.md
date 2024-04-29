# Run IPEX-LLM on Multiple Intel GPUs in pipeline parallel fashion

This example demonstrates how to run IPEX-LLM optimized low-bit model vertically partitioned on two [Intel GPUs](../README.md).

## Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1.1 Install IPEX-LLM

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade ipex-llm[xpu_2.1] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```

### 1.2 Build and install patched version of Intel Extension for PyTorch (IPEX)

```bash
conda activate llm
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.1.10+xpu
git submodule update --init --recursive
git cherry-pick be8ea24078d8a271e53d2946ac533383f7a2aa78
export USE_AOT_DEVLIST='ats-m150,pvc'
python setup.py install
```


> **Important**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

### 2. Run pipeline parallel inference on multiple GPUs
Here, we provide example usages on different models and different hardwares. Please refer to the appropriate script based on your model and device:

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --gpu-num GPU_NUM
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--gpu-num GPU_NUM`: argument defining the number of GPU to use. It is default to be `2`.

#### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<s>[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  Artificial intelligence (AI) is the broader field of research and development aimed at creating machines that can perform tasks that typically require human intelligence,
```