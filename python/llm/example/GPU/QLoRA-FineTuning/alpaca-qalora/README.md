# Alpaca QALoRA Finetuning (experimental support)

This example ports [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) and [QALora](https://github.com/yuhuixu1993/qa-lora/) to BigDL-LLM QALoRA on [Intel GPUs](../../README.md).

### 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install datasets transformers==4.34.0
pip install fire peft==0.5.0
pip install oneccl_bind_pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu # necessary to run distributed finetuning
pip install accelerate==0.23.0
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune

Here, we provide example usages on different hardware. Please refer to the appropriate script based on your device:

#### Finetuning LLaMA2-7B on single Arc A770

```bash
bash finetune_llama2_7b_arc_1_card.sh
```

#### Finetuning LLaMA2-7B on two Arc A770

```bash
bash finetune_llama2_7b_arc_2_card.sh
```

#### Finetuning LLaMA2-7B on single Tile Intel Data Center GPU Max 1550

```bash
bash finetune_llama2_7b_pvc_1550_1_tile.sh
```

**Important: If you fail to complete the whole finetuning process, it is suggested to resume training from a previously saved checkpoint by specifying `resume_from_checkpoint` to the local checkpoint folder as following:**
```bash
python ./alpaca_qalora_finetuning.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qalora-alpaca" \
    --resume_from_checkpoint "./bigdl-qalora-alpaca/checkpoint-1100"
```

### 4. Sample Output
```log
{'loss': 1.8755, 'learning_rate': 9e-05, 'epoch': 0.0}

{'loss': 1.8567, 'learning_rate': 9e-05, 'epoch': 0.01}

{'loss': 1.9291, 'learning_rate': 9e-05, 'epoch': 0.01}

{'loss': 1.82, 'learning_rate': 9e-05, 'epoch': 0.01}

{'loss': 1.827, 'learning_rate': 9e-05, 'epoch': 0.01}

{'loss': 1.8401, 'learning_rate': 9e-05, 'epoch': 0.02}

{'loss': 1.7644, 'learning_rate': 9e-05, 'epoch': 0.02}

{'loss': 1.8418, 'learning_rate': 9e-05, 'epoch': 0.02}                                                              
  1%|█                                                                                                                                                         | 8/1164 [xx:xx<xx:xx:xx, xx s/it]
```
