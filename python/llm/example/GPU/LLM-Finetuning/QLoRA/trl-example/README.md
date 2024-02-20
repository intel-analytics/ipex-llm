# Example of QLoRA Finetuning with BigDL-LLM

This simple example demonstrates how to finetune a llama2-7b model use BigDL-LLM 4bit optimizations using [Intel GPUs](../../../README.md).
Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Finetune llama2-7b using qlora

The `export_merged_model.py` is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py).

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0 datasets
pip install peft==0.5.0
pip install accelerate==0.23.0
pip install bitsandbytes scipy trl
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune model

```
python ./qlora_finetuning.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH
```

#### Sample Output
```log
{'loss': 1.7386, 'learning_rate': 8.888888888888888e-06, 'epoch': 0.19}            
{'loss': 1.9242, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.22}            
{'loss': 1.6819, 'learning_rate': 4.444444444444444e-06, 'epoch': 0.26}            
{'loss': 1.755, 'learning_rate': 2.222222222222222e-06, 'epoch': 0.29}             
{'loss': 1.7455, 'learning_rate': 0.0, 'epoch': 0.32}                              
{'train_runtime': 172.8523, 'train_samples_per_second': 4.628, 'train_steps_per_second': 1.157, 'train_loss': 1.9101631927490235, 'epoch': 0.32}
100%|████████████████████████████████████████████| 200/200 [02:52<00:00,  1.16it/s]
TrainOutput(global_step=200, training_loss=1.9101631927490235, metrics={'train_runtime': 172.8523, 'train_samples_per_second': 4.628, 'train_steps_per_second': 1.157, 'train_loss': 1.9101631927490235, 'epoch': 0.32})
```

### 4. Merge the adapter into the original model

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
