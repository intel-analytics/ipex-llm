# Run IPEX-LLM on Multiple Intel GPUs in Pipeline Parallel Fashion

This example demonstrates how to run IPEX-LLM optimized low-bit model vertically partitioned on multiple [Intel GPUs](../README.md) for Linux users.

## Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Verified Models
- [meta-llama/Llama-2-7b-chat-hf](./run_llama_arc_2_card.sh)
- [meta-llama/Llama-2-13b-chat-hf](./run_llama_arc_2_card.sh)
- [meta-llama/Meta-Llama-3-8B-Instruct](./run_llama_arc_2_card.sh)
- [Qwen/Qwen1.5-7B-Chat](./run_qwen1.5_arc_2_card.sh)
- [Qwen/Qwen1.5-14B-Chat](./run_qwen1.5_arc_2_card.sh)
- [baichuan-inc/Baichuan2-7B-Chat](./run_baichuan2_arc_2_card.sh)
- [baichuan-inc/Baichuan2-13B-Chat](./run_baichuan2_arc_2_card.sh)

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

For optimal performance, it is recommended to set several environment variables. We provide example usages as following:

</details>

<details>
  <summary> Show Llama2 and Llama3 example </summary>

#### Run Llama-2-7b-chat-hf / Llama-2-13b-chat-hf / Meta-Llama-3-8B-Instruct on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Llama2 / Llama3 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
bash run_llama_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Qwen1.5 example </summary>

#### Run Qwen1.5-7B-Chat / Qwen1.5-14B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Qwen1.5 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_qwen1.5_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Baichuan2 example </summary>

#### Run Baichuan2-7B-Chat / Baichuan2-13B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Baichuan2 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_baichuan2_arc_2_card.sh
```

</details>

### 3. Sample Output
#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
First token cost xxxx s and rest tokens cost average xxxx s
-------------------- Prompt --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
-------------------- Output --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She was always asking her parents to take her on trips, but they were always too busy or too tired.

One day, the little girl
```
