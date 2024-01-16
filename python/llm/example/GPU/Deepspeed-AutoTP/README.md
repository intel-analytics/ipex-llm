# Run BigDL-LLM on Multiple Intel GPUs using DeepSpeed AutoTP

This example demonstrates how to run BigDL-LLM optimized low-bit model on multiple [Intel GPUs](../README.md) by leveraging DeepSpeed AutoTP.

## Requirements
To run this example with BigDL-LLM on Intel GPUs, you should install GPU driver and oneAPI Base Toolkit beforehand. See the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more details.

For this particular example, you will need at least two GPUs on your machine.

## Example:

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install BigDL-LLM with PyTorch 2.1 as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install oneccl_bind_pt==2.1.100 -f https://developer.intel.com/ipex-whl-stable-xpu
pip install git+https://github.com/microsoft/DeepSpeed.git@4fc181b0
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@ec33277
pip install mpi4py
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```

### 2. Runtime Configuration
```bash
# Configure oneAPI environment variables
source /opt/intel/oneapi/setvars.sh
```

### 3. Run tensor parallel inference on multiple GPUs
Here, we separate inference process into two stages. First, convert to deepspeed model and apply bigdl-llm optimization on CPU. Then, utilize XPU as DeepSpeed accelerator to inference. In this way, a *X*B model saved in 16-bit will requires approximately 0.5*X* GB total GPU memory in the whole process. For example, if you select to use two GPUs, 0.25*X* GB memory is required per GPU.

Please select the appropriate model size based on the capabilities of your machine.

We provide example usages on different models and different hardwares as following:

- Run LLaMA2-70B on one card of Intel Data Center GPU Max 1550

```
bash run_llama2_70b_pvc_1550_1_card.sh
```

> **Note**: You could change `ZE_AFFINITY_MASK` and `NUM_GPUS` according to your requirements.

- Run Vicuna-33B on two Intel Arc A770

```
bash run_vicuna_33b_arc_2_card.sh
```

> **Note**: You could change `NUM_GPUS` to the number of GPUs you have on your machine.


### Known Issue
- In our example scripts, tcmalloc is enabled through `export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}` which speed up inference, but this may raise `munmap_chunk(): invalid pointer` error after finishing inference.
