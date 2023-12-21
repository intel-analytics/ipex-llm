# Alpaca QLoRA Finetuning of Mixtral-8x7B

This example ports [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) to BigDL-LLM to showcase how to finetune [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) on [Intel Data Center GPU](../../README.md). 

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

## 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install datasets transformers==4.36.1
pip install fire peft==0.5.0
pip install accelerate==0.23.0
```

## 2. Configures OneAPI environment variables
```bash
# intel_extension_for_pytorch==2.1.10+xpu requires oneAPI 2024.0
source /opt/intel/oneapi/setvars.sh
```

## 3. Finetune on Intel Data Center GPU
```bash
bash finetune_mixtral_8x7b_pvc_1550_1_tile.sh
```

## 4. (Optional) Resume Training
If you fail to complete the whole finetuning process, it is suggested to resume training from a previously saved checkpoint by specifying `resume_from_checkpoint` to the local checkpoint folder as following:**
```bash
python ./alpaca_qlora_finetuning.py \
    --base_model "mistralai/Mixtral-8x7B-v0.1" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca" \
    --resume_from_checkpoint "./bigdl-qlora-alpaca/checkpoint-1100"
```

## 5. Sample Loss Output
```log
{'loss': 1.6294, 'learning_rate': 0.00019999998724475014, 'epoch': 0.0}                                                                               
{'loss': 1.5134, 'learning_rate': 0.00019999994897900374, 'epoch': 0.0}                                                                               
{'loss': 1.1062, 'learning_rate': 0.0001999998852027706, 'epoch': 0.0}                                                                                
{'loss': 0.9595, 'learning_rate': 0.00019999979591606696, 'epoch': 0.0}                                                                               
{'loss': 1.0657, 'learning_rate': 0.00019999968111891563, 'epoch': 0.0}                                                                               
{'loss': 1.2788, 'learning_rate': 0.0001999995408113459, 'epoch': 0.0}                                                                                
{'loss': 1.1196, 'learning_rate': 0.00019999937499339354, 'epoch': 0.0}                                                                               
{'loss': 0.9707, 'learning_rate': 0.0001999991836651008, 'epoch': 0.0}                                                                                
{'loss': 0.9065, 'learning_rate': 0.0001999989668265166, 'epoch': 0.0}                                                                                
{'loss': 0.8428, 'learning_rate': 0.00019999872447769624, 'epoch': 0.0}                                                                               
{'loss': 0.9477, 'learning_rate': 0.00019999845661870146, 'epoch': 0.0}                                                                               
{'loss': 0.8384, 'learning_rate': 0.00019999816324960064, 'epoch': 0.0}                                                                               
  0%|▏                                                                                                           | 12/6220 [xx:xx<xx:xx:xx, xx s/it]
```

## 6. Merge the adapter into the original model
```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
