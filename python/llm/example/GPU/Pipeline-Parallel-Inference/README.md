# Run IPEX-LLM on Multiple Intel GPUs in Pipeline Parallel Fashion

This example demonstrates how to run IPEX-LLM optimized low-bit model vertically partitioned on multiple [Intel GPUs](../README.md) for Linux users.

## Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Verified Models
- [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Example: Run pipeline parallel inference on multiple GPUs

### 0. Prerequisites

Please visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), follow [Install Intel GPU Driver](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-intel-gpu-driver) and [Install oneAPI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

### 1. Installation

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Run pipeline parallel inference on multiple GPUs

For optimal performance, it is recommended to set several environment variables. We provide example usage as following:

- Run Llama-2-13b-chat-hf on two Intel Arc A770

```bash
bash run_llama2_13b_arc_2_card.sh
```

> **Note**: You could change `NUM_GPUS` to the number of GPUs you have on your machine.

#### Sample Output
##### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
First token cost xxxx s and rest tokens cost average xxxx s
-------------------- Prompt --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
-------------------- Output --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She was always asking her parents to take her on trips, but they were always too busy or too tired.

One day, the little girl
```