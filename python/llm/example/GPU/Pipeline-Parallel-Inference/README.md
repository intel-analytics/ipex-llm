# Run BigDL-LLM on Multiple Intel GPUs in pipeline parallel fashion

This example demonstrates how to run BigDL-LLM optimized low-bit model vertically partitioned on two [Intel GPUs](../README.md).

## Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1.1 Install BigDL-LLM

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```

### 1.2 Build and install patched version of Intel Extension for PyTorch (IPEX)

```bash
conda activate llm
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/yangw1234/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout fix-cache
git submodule update --init --recursive
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=1
export USE_NUMA=0
export USE_CUDA=0
export USE_XETLA=OFF
export BUILD_WITH_CPU=OFF
export USE_AOT_DEVLIST='ats-m150,pvc'
python setup.py install
```


> **Important**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

### 2. Run tensor parallel inference on multiple GPUs
Here, we provide example usages on different models and different hardwares. Please refer to the appropriate script based on your model and device:

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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

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