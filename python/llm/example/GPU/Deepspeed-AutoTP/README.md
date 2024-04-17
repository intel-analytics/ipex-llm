# Run IPEX-LLM on Multiple Intel GPUs using DeepSpeed AutoTP

This example demonstrates how to run IPEX-LLM optimized low-bit model on multiple [Intel GPUs](../README.md) by leveraging DeepSpeed AutoTP.

## Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
pip install git+https://github.com/microsoft/DeepSpeed.git@ed8aed5
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@0eb734b
pip install mpi4py
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```
> **Important**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

### 2. Run tensor parallel inference on multiple GPUs

Here, we separate inference process into two stages. First, convert to deepspeed model and apply ipex-llm optimization on CPU. Then, utilize XPU as DeepSpeed accelerator to inference. In this way, a *X*B model saved in 16-bit will requires approximately 0.5*X* GB total GPU memory in the whole process. For example, if you select to use two GPUs, 0.25*X* GB memory is required per GPU.

Please select the appropriate model size based on the capabilities of your machine.

We provide example usages on different models and different hardwares as following:

- Run LLaMA2-70B on one card of Intel Data Center GPU Max 1550

```bash
bash run_llama2_70b_pvc_1550_1_card.sh
```

> **Note**: You could change `ZE_AFFINITY_MASK` and `NUM_GPUS` according to your requirements. And you could also specify other low bit optimizations through `--low-bit`.

- Run Vicuna-33B on two Intel Arc A770

```bash
bash run_vicuna_33b_arc_2_card.sh
```

> **Note**: You could change `NUM_GPUS` to the number of GPUs you have on your machine. And you could also specify other low bit optimizations through `--low-bit`.

- Run Mistral-7B-Instruct on two cards of Intel Data Center GPU Flex

```bash
bash run_mistral_7b_instruct_flex_2_card.sh
```

> **Note**: You could change `NUM_GPUS` to the number of GPUs you have on your machine. And you could also specify other low bit optimizations through `--low-bit`.

### 3. Sample Output

```bash
[0] Inference time of generating 32 tokens: xxx s, average token latency is xxx ms/token.
[0] -------------------- Prompt --------------------
[0] Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
[0] -------------------- Output --------------------
[0] Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She was a curious girl, and she loved to learn new things.
[0] 
[0] One day, she decided to go on a journey to find the legendary
```

**Important**: The first token latency is much larger than rest token latency, you could use [our benchmark tool](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/dev/benchmark/README.md) to obtain more details about first and rest token latency.

### Known Issue

- In our example scripts, tcmalloc is enabled through `export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}` which speed up inference, but this may raise `munmap_chunk(): invalid pointer` error after finishing inference.
