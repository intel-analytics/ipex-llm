# Distributed Finetuning LLAMA Using QLoRA (experimental support)

This example demonstrates how to finetune a llama2-7b model using Big-LLM 4bit optimizations on [Intel CPUs](../README.md).


## Example: Distributed Finetune llama2-7b using QLoRA


### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[all]
pip install transformers==4.34.0
pip install peft==0.5.0
pip install datasets
pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable
pip install fair
```

### 2. Finetune model
This example is to finetune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [Cleaned alpaca data](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json)

If the machine memory is not enough, you can try to set `use_gradient_checkpointing=True` in [here](https://github.com/intel-analytics/BigDL/blob/1747ffe60019567482b6976a24b05079274e7fc8/python/llm/example/CPU/QLoRA-FineTuning/qlora_finetuning_cpu.py#L53C6-L53C6).

And remember to use `bigdl-llm-init` before you start finetuning, which can accelerate the job.

```
source ${conda_env}/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
source bigdl-llm-init -t
# add ./hosts and update confs in mpirun_qlora.sh
bash mpirun_qlora.sh
```

