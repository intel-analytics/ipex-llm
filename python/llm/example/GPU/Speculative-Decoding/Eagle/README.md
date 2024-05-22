# Eagle - Speculative Sampling using IPEX-LLM on Intel GPUs
IPEX-LLM supports EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) which is a speculative sampling method that improves text generation speed.

See [here](https://arxiv.org/abs/2401.15077) to view the paper and [here](https://github.com/SafeAILab/EAGLE) for more info on EAGLE code.

## Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### Verified Hardware Platforms

- Intel Data Center GPU Max Series

## Example - EAGLE Speculative Sampling with IPEX-LLM on MT-bench
In this example, we run inference for a Llama2 model to showcase the speed of EAGLE with IPEX-LLM on MT-bench on Intel GPUs.

### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install eagle-llm
pip install -r requirements.txt
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm
# below command will use pip to install the Intel oneAPI Base Toolkit 2024.0
pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install eagle-llm
pip install -r requirements.txt
```

### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

### 4. Running Example
You can test the speed of EAGLE speculative sampling with ipex-llm on MT-bench using the following command.
```bash
python -m evaluation.gen_ea_answer_llama2chat\
                 --ea-model-path [path of EAGLE weight]\
                 --base-model-path [path of the original model]\
                 --enable-ipex-llm\
```
Please refer to [here](https://github.com/SafeAILab/EAGLE#eagle-weights) for the complete list of available EAGLE weights.

The above command will generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the speed.
```bash
python -m evaluation.speed\
                 --base-model-path [path of the original model]\
                 --jsonl-file [pathname of the .jsonl file]\
```


