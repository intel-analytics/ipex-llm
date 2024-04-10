# GaLore Finetuning with IPEX-LLM

This example ports [GaLore on Llama2 7B](https://github.com/geronimi73/3090_shorts/blob/main/nb_galore_llama2-7b.ipynb) to IPEX-LLM on [Intel GPU](../../../README.md).

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n heyang-galore python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install galore-torch
pip install accelerate==0.28.0
pip install bitsandbytes==0.43.0
pip install datasets==2.18.0
pip install transformers==4.39.1
pip install trl==0.8.1
pip install fire
```

### 2. GaLore Finetune

Currently, GaLore only supports local fine-tuning, and here is how to fine-tune Llama2 7B on an Intel Max GPU server:

```bash
# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
python galore_finetuning.py # optional parameters as below
```

Optional parameters for `galore_finetuning.py`:

**--repo-id-or-model-path** : default to `meta-llama/Llama-2-7b-hf`, and you can also specify your local model path.

**--data-path** : default to `g-ronimo/oasst2_top4k_en`, and you can also specify your local datal path.

**--output-dir** : default to `./ipex-llm-galore` to save fine-tuned model, and you can change if needed.

### 3. Sample Output
```log
......
find instruction key `<|im_start|>user` in the following instance: <|im_start|> user
According to noted psychologist Professor Jordan Peterson, intelligence and wisdom are not only not the same thing, they are not even correlated.

Consequently, I would argue, anyone who undertook to create and market an "artificial intelligence” program should also interest themselves in being able to market “artificial wisdom.”

But, you will rejoin, “Anybody who claims to be wise is vulnerable to the charge of sophistry.”  This is correct.  However, I propose a solution to that problem, by inverting the question.  Do not ask, "who is systematically wise?" but rather, “Who is systematically opposed by, or systematically ignored by, people who are not wise?”

Of course, that changes the issue to, “Who is most prominently ‘not wise’ in American society?”  I put it to you that that question reduces down to “who is prominently superficial and negative?”  Negativity is not wise because it is equivalent to ingratitude - the opposite of counting your blessings - and that (as Dennis Prager noted in his book, “Happiness is a Serious Problem”) This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.
{'loss': 1.0935, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.32}
{'loss': 1.168, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.33}
{'loss': 1.0114, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.33}
{'loss': 1.2606, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.34}
{'loss': 1.1006, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.34}
{'loss': 0.9969, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.34}
 11%|████████                                                              | 86/750 [xx:xx<x:xx:xx, xx.xxs/it]
 ......

```
