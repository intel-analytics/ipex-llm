# Run BigDL-LLM on Multiple Intel GPUs using DeepSpeed AutoTP

This example demonstrates how to run BigDL-LLM optimized low-bit model on multiple [Intel GPUs](../README.md) by leveraging DeepSpeed AutoTP.

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install oneccl_bind_pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu
pip install git+https://github.com/microsoft/DeepSpeed.git@78c518e
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@ec33277
pip install mpi4py
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run tensor parallel inference on multiple GPUs
You many want to change some of the parameters in the script such as `NUM_GPUS`` to the number of GPUs you have on your machine.

```
bash run.sh
```
