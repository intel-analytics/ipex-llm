# Finetune (QLoRA)

We also support finetuning LLMs (large language models) using QLoRA with BigDL-LLM 4bit optimizations on Intel GPUs.

```eval_rst
.. note::

   Currently, only Hugging Face Transformers models are supported running QLoRA finetuning.
```

To help you better understand the finetuning process, here we use model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example.

**Make sure you have prepared environment following instructions [here](../install_gpu.html).**

```eval_rst
.. note::

   If you are using an older version of ``bigdl-llm`` (specifically, older than 2.5.0b20240104), you need to manually add ``import intel_extension_for_pytorch as ipex`` at the beginning of your code.
```

First, load model using `transformers`-style API and **set it to `to('xpu')`**. We specify `load_in_low_bit="nf4"` here to apply 4-bit NormalFloat optimization. According to the [QLoRA paper](https://arxiv.org/pdf/2305.14314.pdf), using `"nf4"` could yield better model quality than `"int4"`.

```python
from bigdl.llm.transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                             load_in_low_bit="nf4",
                                             optimize_model=False,
                                             torch_dtype=torch.float16,
                                             modules_to_not_convert=["lm_head"],)
model = model.to('xpu')
```

Then, we have to apply some preprocessing to the model to prepare it for training.
```python
from bigdl.llm.transformers.qlora import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

Next, we can obtain a Peft model from the optimized model and a configuration object containing the parameters as follows:
```python
from bigdl.llm.transformers.qlora import get_peft_model
from peft import LoraConfig
config = LoraConfig(r=8, 
                    lora_alpha=32, 
                    target_modules=["q_proj", "k_proj", "v_proj"], 
                    lora_dropout=0.05, 
                    bias="none", 
                    task_type="CAUSAL_LM")
model = get_peft_model(model, config)
```

```eval_rst
.. important::

   Instead of ``from peft import prepare_model_for_kbit_training, get_peft_model`` as we did for regular QLoRA using bitandbytes and cuda, we import them from ``bigdl.llm.transformers.qlora`` here to get a BigDL-LLM compatible Peft model. And the rest is just the same as regular LoRA finetuning process using ``peft``.
```

```eval_rst
.. seealso::

   See the complete examples `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/GPU>`_
```
