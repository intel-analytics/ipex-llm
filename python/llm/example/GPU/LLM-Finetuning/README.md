# Running LLM Finetuning using IPEX-LLM on Intel GPU

This folder contains examples of running different training mode with IPEX-LLM on Intel GPU:

- [LoRA](LoRA): examples of running LoRA finetuning
- [QLoRA](QLoRA): examples of running QLoRA finetuning
- [QA-LoRA](QA-LoRA): examples of running QA-LoRA finetuning
- [ReLora](ReLora): examples of running ReLora finetuning
- [DPO](DPO): examples of running DPO finetuning
- [common](common): common templates and utility classes in finetuning examples
- [HF-PEFT](HF-PEFT): run finetuning on Intel GPU using Hugging Face PEFT code without modification
- [axolotl](axolotl): LLM finetuning on Intel GPU using axolotl without writing code

## Verified Models

| Model      | Finetune mode                                                   | Frameworks Support |
|------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| LLaMA 2/3    |   [LoRA](LoRA), [QLoRA](QLoRA), [QA-LoRA](QA-LoRA), [ReLora](ReLora)  | [HF-PEFT](HF-PEFT), [axolotl](axolotl) |
| Mistral | [LoRA](DPO), [QLoRA](DPO) | [DPO](DPO) |
| ChatGLM 3 | [LoRA](LoRA/chatglm_finetune#lora-fine-tuning-on-chatglm3-6b-with-ipex-llm), [QLoRA](QLoRA/alpaca-qlora#3-qlora-finetune) | HF-PEFT |
| Qwen-1.5 | [QLoRA](QLoRA/alpaca-qlora#3-qlora-finetune) | HF-PEFT |
| Baichuan2 | [QLoRA](QLoRA/alpaca-qlora#3-qlora-finetune) | HF-PEFT |

## Troubleshooting
- If you fail to finetune on multi cards because of following error message:
  ```bash
  RuntimeError: oneCCL: comm_selector.cpp:57 create_comm_impl: EXCEPTION: ze_data was not initialized
  ```
  Please try `sudo apt install level-zero-dev` to fix it.

- Please raise the system open file limit using `ulimit -n 1048576`. Otherwise, there may exist error `Too many open files`.

- If application raise `wandb.errors.UsageError: api_key not configured (no-tty)`. Please login wandb or disable wandb login with this command:

```bash
export WANDB_MODE=offline
```

- If application raise Hugging Face related errors, i.e., `NewConnectionError` or `Failed to download` etc. Please download models and datasets, set model and data path, then set `HF_HUB_OFFLINE` with this command:

```bash
export HF_HUB_OFFLINE=1
```
