# Run BigDL-LLM on Multiple Intel GPUs using DeepSpeed AutoTP

This example demonstrates how to run BigDL-LLM optimized low-bit model on multiple [Intel GPUs](../README.md) by leveraging DeepSpeed AutoTP.

## Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install oneccl_bind_pt==2.1.100 -f https://developer.intel.com/ipex-whl-stable-xpu
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
pip install git+https://github.com/microsoft/DeepSpeed.git@4fc181b0
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@ec33277
pip install mpi4py
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```
> **Important**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

### 2. Run tensor parallel inference on multiple GPUs
Here, we provide example usages on different models and different hardwares. Please refer to the appropriate script based on your model and device:

#### Llama2 series
<details><summary>Show LLaMA2-70B example</summary>
Run LLaMA2-70B on four Intel Data Center GPU Max 1550

```
bash run_llama2_70b_pvc_1550_4_card.sh
```
</details>

> **Note**:If you may want to select only part of GPUs on your machine, please change `ZE_AFFINITY_MASK` and `NUM_GPUS` to your prefer value.
