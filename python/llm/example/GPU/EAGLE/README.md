# running EAGLE on MT-bench using IPEX-LLM on Intel GPUs
You can use IPEX-LLM to run inference for any PyTorch (including EAGLE) model on Intel GPUs. This directory contains example scripts to help you test the speed of EAGLE with IPEX-LLM on MT-bench.
See the [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://github.com/SafeAILab/EAGLE) for the latest on EAGLE.

## Verified Hardware Platforms

- Intel Data Center GPU Max Series

## Recommended Requirements
To apply Intel GPU acceleration, there’re several steps for tools installation and environment preparation. See the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details.

## Best Known Configuration on Linux

For optimal performance on Intel Data Center GPU Max Series, it is recommended to set several environment variables.
```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
## Setup & Installation
```bash
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install eagle-llm
pip install -r requirements.txt
source /opt/intel/oneapi/setvars.sh 
```

## Evaluation
You can test the speed of EAGLE with ipex-llm on MT-bench using the following command.
```bash
python -m evaluation.gen_ea_answer_llama2chat\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
                 --enable-ipex-llm\
```
If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m evaluation.gen_baseline_answer_llama2chat\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
                 --enable-ipex-llm\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.
```bash
python -m evaluation.speed\
                 --base-model-path [path of the original model]\
                 --jsonl-file-base [pathname of the base .jsonl file]\
                 --jsonl-file [pathname of the .jsonl file]\
```

## EAGLE Weights

| Base Model  | EAGLE on Hugging Face  | \# EAGLE Parameters | Base Model  | EAGLE on Hugging Face  | \# EAGLE Parameters |
|------|------|------|------|------|------|
| Vicuna-7B-v1.3 | [yuhuili/EAGLE-Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3) | 0.24B | LLaMA2-Chat 7B | [yuhuili/EAGLE-llama2-chat-7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B) | 0.24B |
| Vicuna-13B-v1.3 | [yuhuili/EAGLE-Vicuna-13B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-13B-v1.3) | 0.37B | LLaMA2-Chat 13B | [yuhuili/EAGLE-llama2-chat-13B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-13B) | 0.37B |
| Vicuna-33B-v1.3 | [yuhuili/EAGLE-Vicuna-33B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-33B-v1.3)| 0.56B | LLaMA2-Chat 70B| [yuhuili/EAGLE-llama2-chat-70B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-70B)| 0.99B |
| Mixtral-8x7B-Instruct-v0.1 | [yuhuili/EAGLE-mixtral-instruct-8x7B](https://huggingface.co/yuhuili/EAGLE-mixtral-instruct-8x7B)| 0.28B |


