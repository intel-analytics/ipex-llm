# EAGLE - Speculative Sampling using IPEX-LLM on Intel GPUs
In this directory, you will find the examples on how IPEX-LLM accelerate inference with speculative sampling using EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), a speculative sampling method that improves text generation speed) on Intel GPUs. See [here](https://arxiv.org/abs/2401.15077) to view the paper and [here](https://github.com/SafeAILab/EAGLE) for more info on EAGLE code.

## Requirements
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation. See the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more details.

Step 1, only Linux system is supported now, Ubuntu 22.04 is prefered.

Step 2, please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
> **Note**: IPEX 2.1.10+xpu requires Intel GPU Driver version >= stable_775_20_20231219.

Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
> **Note**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0.

### Verified Hardware Platforms

- Intel Data Center GPU Max Series
- Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series

## Example - EAGLE-2 Speculative Sampling with IPEX-LLM on MT-bench
In this example, we run inference for a Llama2 model to showcase the speed of EAGLE with IPEX-LLM on MT-bench data on Intel GPUs.
We use EAGLE-2 which have better performance than EAGLE-1

### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
git clone https://github.com/SafeAILab/EAGLE.git
cd EAGLE
pip install -r requirements.txt
pip install -e .
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
git clone https://github.com/SafeAILab/EAGLE.git
cd EAGLE
pip install -r requirements.txt
pip install -e .
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

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>
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
python -m evaluation.gen_ea_answer_llama2chat_e2_ipex_optimize\
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


