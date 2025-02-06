# Simple Example of DPO Finetuning with IPEX-LLM

This simple example demonstrates how to finetune a Mistral-7B model use IPEX-LLM 4bit optimizations using [Intel GPUs](../../README.md).
Note, this example is just used for illustrating related usage.

## 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

## Example: Finetune Mistral-7b using DPO

This example is ported from [Fine_tune_a_Mistral_7b_model_with_DPO](https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb).

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.45.0 "trl<0.12.0" datasets
pip install peft==0.10.0
pip install bitsandbytes==0.45.1
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune model

```
python ./dpo_finetuning.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --gradient-checkpointing
```
> Note: The final LoRA weights and configurations are saved to './outputs' by default. You could also change the output path through specifying `--output-path`.

#### Sample Output
```log
trainable params: 41,943,040 || all params: 4,012,134,400 || trainable%: 1.0454046604221434
{'loss': 0.6931, 'learning_rate': 5.000000000000001e-07, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/rejected': -271.842041015625, 'logps/chosen': -146.93634033203125, 'logits/rejected': -2.9851596355438232, 'logits/chosen': -2.98481822013855, 'epoch': 0.0}
{'loss': 0.6931, 'learning_rate': 1.0000000000000002e-06, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/rejected': -248.09817504882812, 'logps/chosen': -259.561767578125, 'logits/rejected': -2.967536449432373, 'logits/chosen': -2.951939582824707, 'epoch': 0.0}
{'loss': 0.7058, 'learning_rate': 1.5e-06, 'rewards/chosen': -0.006700039375573397, 'rewards/rejected': 0.016817521303892136, 'rewards/accuracies': 0.4375, 'rewards/margins': -0.023517560213804245, 'logps/rejected': -183.52743530273438, 'logps/chosen': -122.3787841796875, 'logits/rejected': -2.948030471801758, 'logits/chosen': -2.9321558475494385, 'epoch': 0.0}
{'loss': 0.6912, 'learning_rate': 2.0000000000000003e-06, 'rewards/chosen': 0.0014888052828609943, 'rewards/rejected': -0.004842948634177446, 'rewards/accuracies': 0.625, 'rewards/margins': 0.006331752985715866, 'logps/rejected': -234.07257080078125, 'logps/chosen': -181.22940063476562, 'logits/rejected': -2.938673496246338, 'logits/chosen': -2.9304277896881104, 'epoch': 0.0}
{'loss': 0.6958, 'learning_rate': 2.5e-06, 'rewards/chosen': -0.001946449396200478, 'rewards/rejected': 0.0025150063447654247, 'rewards/accuracies': 0.5625, 'rewards/margins': -0.004461456090211868, 'logps/rejected': -263.15106201171875, 'logps/chosen': -242.25759887695312, 'logits/rejected': -2.931898832321167, 'logits/chosen': -2.9180212020874023, 'epoch': 0.01}
{'loss': 0.6714, 'learning_rate': 3e-06, 'rewards/chosen': 0.002834760583937168, 'rewards/rejected': -0.043302297592163086, 'rewards/accuracies': 0.625, 'rewards/margins': 0.04613706097006798, 'logps/rejected': -269.76953125, 'logps/chosen': -175.4458465576172, 'logits/rejected': -2.863767147064209, 'logits/chosen': -2.813715696334839, 'epoch': 0.01}
{'loss': 0.6773, 'learning_rate': 3.5000000000000004e-06, 'rewards/chosen': -0.000818049069494009, 'rewards/rejected': -0.03519792854785919, 'rewards/accuracies': 0.6875, 'rewards/margins': 0.034379877150058746, 'logps/rejected': -307.48388671875, 'logps/chosen': -258.1222839355469, 'logits/rejected': -2.93851900100708, 'logits/chosen': -2.845832347869873, 'epoch': 0.01}
```

### 4. Merge the adapter into the original model

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs --output_path ./outputs/merged-model
```

Then you can use `./outputs/merged-model` as a normal huggingface transformer model to do inference.
