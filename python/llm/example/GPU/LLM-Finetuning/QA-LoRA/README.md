# QA-LoRA Finetuning with IPEX-LLM

This example ports [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) to IPEX-LLM (using [QA-LoRA](https://arxiv.org/abs/2309.14717) algorithm) on [Intel GPU](../../README.md).

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.34.0 datasets
pip install fire peft==0.5.0
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ # necessary to run distributed finetuning
pip install accelerate==0.23.0
pip install bitsandbytes scipy
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. QA-LoRA Finetune

Here, we provide example usages on different hardware. Please refer to the appropriate script based on your device:

##### Finetuning LLaMA2-7B on single Arc A770

```bash
bash qalora_finetune_llama2_7b_arc_1_card.sh
```

##### Finetuning LLaMA2-7B on two Arc A770

```bash
bash qalora_finetune_llama2_7b_arc_2_card.sh
```

##### Finetuning LLaMA2-7B on single tile of Intel Data Center GPU Max 1550

```bash
bash qalora_finetune_llama2_7b_pvc_1550_1_tile.sh
```

### 4. (Optional) Resume Training
**If you fail to complete the whole finetuning process, it is suggested to resume training from a previously saved checkpoint by specifying `resume_from_checkpoint` to the local checkpoint folder as following:**
```bash
python ./alpaca_qalora_finetuning.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./ipex-llm-qlora-alpaca" \
    --resume_from_checkpoint "./ipex-llm-qlora-alpaca/checkpoint-1100"
```

### 5. Sample Output
```log
{'loss': 1.9231, 'learning_rate': 2.9999945367033285e-05, 'epoch': 0.0}                                                                                                                            
{'loss': 1.8622, 'learning_rate': 2.9999781468531096e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.9043, 'learning_rate': 2.9999508305687345e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8967, 'learning_rate': 2.999912588049185e-05, 'epoch': 0.01}                                                                                                                            
{'loss': 1.9658, 'learning_rate': 2.9998634195730358e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8386, 'learning_rate': 2.9998033254984483e-05, 'epoch': 0.02}                                                                                                                           
{'loss': 1.809, 'learning_rate': 2.999732306263172e-05, 'epoch': 0.02}                                                                                                                             
{'loss': 1.8552, 'learning_rate': 2.9996503623845395e-05, 'epoch': 0.02}                                                                                                                           
  1%|█                                                                                                                                                         | 8/1164 [xx:xx<xx:xx:xx, xx s/it]
```

### 6. Merge the adapter into the original model
```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.

### 7. Troubleshooting
Please refer to [here](../README.md#troubleshooting) for solutions of common issues during finetuning.
