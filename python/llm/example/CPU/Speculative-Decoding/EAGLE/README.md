# EAGLE - Speculative Sampling using IPEX-LLM on Intel CPUs
In this directory, you will find the examples on how IPEX-LLM accelerate inference with speculative sampling using EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), a speculative sampling method that improves text generation speed) on Intel CPUs. See [here](https://arxiv.org/abs/2401.15077) to view the paper and [here](https://github.com/SafeAILab/EAGLE) for more info on EAGLE code.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../README.md#system-support) for more information. Make sure you have installed `ipex-llm` before:

## Example - EAGLE Speculative Sampling with IPEX-LLM on MT-bench
In this example, we run inference for a Llama2 model to showcase the speed of EAGLE with IPEX-LLM on MT-bench data on Intel CPUs.

### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:
```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==2.1.0 
pip install -r requirements.txt
pip install transformers==4.36.2
pip install gradio==3.50.2
pip install eagle-llm
```

### 2. Configures IPEX-LLM environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.
```bash
# set IPEX-LLM env variables
source ipex-llm-init

```
### 3. Running Example
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

