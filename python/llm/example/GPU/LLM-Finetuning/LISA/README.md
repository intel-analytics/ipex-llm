# LISA Finetuning with IPEX-LLM

This example refers to [LISA with LMFLow's DynamicLayerActivationCallback Class](https://github.com/OptimalScale/LMFlow/blob/f3b3b007ea526009172c355e9d52ffa146b9dc0c/src/lmflow/pipeline/finetuner.py#L301), and adds [LISA fintuning](https://arxiv.org/abs/2403.17919) to IPEX-LLM on [Intel GPU](../../../GPU/README.md), based on [LORA finetuning with IPEX-LLM](../LoRA/alpaca_lora_finetuning.py).

### 0. Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../GPU/README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.45.0 "trl<0.12.0" datasets
pip install bitsandbytes==0.45.1 scipy fire
```

### 2. LISA Finetune

```bash
# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
python ./lisa_finetuning.py \
    --micro_batch_size 8 \
    --batch_size 128 \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./ipex-llm-lisa-alpaca" \
    --gradient_checkpointing True \
    --lisa_activated_layers 1 \
    --lisa_interval_steps 20
```

Optional parameters for `lisa_finetuning.py`:

**--repo-id-or-model-path** : default to `meta-llama/Llama-2-7b-hf`, and you can also specify your local model path.

**--data-path** : default to `yahma/alpaca-cleaned`, and you can also specify your local datal path, while note that changing to the other datasets will introduce code modification effort for yourself.

**--output-dir** : default to `./ipex-llm-lisa-alpaca` to save fine-tuned model, and you can change if needed.

**--lisa_activated_layers** :  the number of self-attention layers randomly selected to activate.

**lisa_interval_steps** : the number of interval steps to switch active layers.

### 3. Sample Output

```log
......
{'loss': 1.8391, 'learning_rate': 1.9967238104745695e-05, 'epoch': 0.03}
{'loss': 1.8242, 'learning_rate': 1.9869167087338908e-05, 'epoch': 0.05}
  5%|██████▉                                                        | 20/388 [xx:xx<x:xx:xx,  x.xxs/it]
Activating layers at indices: [10] for the next steps.
{'loss': 1.8128, 'learning_rate': 1.9706429546259592e-05, 'epoch': 0.08}
{'loss': 1.775, 'learning_rate': 1.9480091799562706e-05, 'epoch': 0.1}
 10%|██████████████                                                  | 40/388 [xx:xx<xx:xx,  x.xxs/it]
Activating layers at indices: [30] for the next steps.
{'loss': 1.7669, 'learning_rate': 1.9191636897958123e-05, 'epoch': 0.13}
{'loss': 1.7749, 'learning_rate': 1.8842954907300236e-05, 'epoch': 0.15}
 15%|█████████████████████                                           | 60/388 [xx:xx<xx:xx,  x.xxs/it]
Activating layers at indices: [26] for the next steps.
{'loss': 1.7735, 'learning_rate': 1.8436330524160048e-05, 'epoch': 0.18}
{'loss': 1.7199, 'learning_rate': 1.797442810562721e-05, 'epoch': 0.21}
 21%|████████████████████████████                                    | 80/388 [xx:xx<xx:xx,  x.xxs/it]
Activating layers at indices: [17] for the next steps.
{'loss': 1.7328, 'learning_rate': 1.7460274211432463e-05, 'epoch': 0.23}
 25%|█████████████████████████████████▋                             | 96/388 [xx:xx<xx:xx,  x.xxs/it]
 ......

```
