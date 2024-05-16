# Serve IPEX-LLM on Multiple Intel GPUs in multi-stage pipeline parallel fashion

This example demonstrates how to run IPEX-LLM serving on multiple [Intel GPUs](../README.md) with Pipeline Parallel.

## Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Run pipeline parallel serving on multiple GPUs

```bash
# Need to set MODEL_PATH in run.sh first
bash run.sh
```

