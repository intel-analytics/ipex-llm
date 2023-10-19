# GPU Supports

## Run LLMs

Apart from the significant acceleration capabilites on Intel CPUs, BigDL-LLM also supports optimizations and acceleration for running LLMs (large language models) on Intel GPUs. With BigDL-LLM, PyTorch models (in FP16/BF16/FP32) can be optimized with low-bit quantizations (supported precisions include INT4, INT5, INT8, etc).

Compared with running on Intel CPUs, some additional operations are required on Intel GPUs. To help you better understand the process, here we use a popular model [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as an example.

**Make sure you have prepared environment following instructions [here](../install_gpu.html). First of all, you need to import `intel_extension_for_pytorch` to run on Intel GPUs**:

```python
import intel_extension_for_pytorch as ipex
```

### Load and Optimize Model

You could choose to use [PyTorch API](./optimize_model.html) or [`transformers`-style API](./transformers_style_api.html) on Intel GPUs based on your requirements.

**Once you have the model with BigDL-LLM low bit optimization, set it to `to('xpu')`**.

```eval_rst
.. tabs::

   .. tab:: PyTorch API

      You could optimize any PyTorch model with "one-line code change", and the loading and optimizing process on Intel GPUs maybe as follows:
      
      .. code-block:: python

         # Take Llama-2-7b-chat-hf as an example
         import intel_extension_for_pytorch as ipex
         from transformers import LlamaForCausalLM
         from bigdl.llm import optimize_model

         model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)
         model = optimize_model(model) # With only one line to enable BigDL-LLM INT4 optimization

         model = model.to('xpu') # Important after obtaining the optimized model

      Especially, if you have saved the optimized model following setps `here <https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/KeyFeatures/optimize_model.html#save>`_, the loading process on Intel GPUs maybe as follows:

      .. code-block:: python

         import intel_extension_for_pytorch as ipex
         from transformers import LlamaForCausalLM
         from bigdl.llm.optimize import low_memory_init, load_low_bit

         saved_dir='./llama-2-bigdl-llm-4-bit'
         with low_memory_init(): # Fast and low cost by loading model on meta device
            model = LlamaForCausalLM.from_pretrained(saved_dir,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
         model = load_low_bit(model, saved_dir) # Load the optimized model

         model = model.to('xpu') # Important after obtaining the optimized model

   .. tab:: ``transformers``-style API

      You could run any Hugging Face Transformers model with ``transformers``-style API, and the loading and optimizing process on Intel GPUs maybe as follows:
      
      .. code-block:: python

         # Take Llama-2-7b-chat-hf as an example
         import intel_extension_for_pytorch as ipex
         from bigdl.llm.transformers import AutoModelForCausalLM

         # Load model in 4 bit, which convert the relevant layers in the model into INT4 format
         model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', load_in_4bit=True)

         model = model.to('xpu') # Important after obtaining the optimized model

      Especially, if you have saved the optimized model following setps `here <https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/KeyFeatures/hugging_face_format.html#save-load>`_, the loading process on Intel GPUs maybe as follows:

      .. code-block:: python

         import intel_extension_for_pytorch as ipex
         from bigdl.llm.transformers import AutoModelForCausalLM

         saved_dir='./llama-2-bigdl-llm-4-bit'
         model = AutoModelForCausalLM.load_low_bit(saved_dir) # Load the optimized model

         model = model.to('xpu') # Important after obtaining the optimized model
```

### Run Optimized Model

You could then do inference using the optimized model on Intel GPUs almostly the same as on CPUs. **The only difference is to set `to('xpu')` for input tensors and `.cpu()` for output tensors.**

Continuing with the [example of Llama-2-7b-chat-hf](#load-and-optimize-model), running as follows:
```python
import torch

with torch.inference_mode():
   prompt = 'Q: What is CPU?\nA:'
   input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu') # With .to('xpu') specifically for inference on Intel GPUs
   output = model.generate(input_ids, max_new_tokens=32)
   torch.xpu.synchronize()
   output = output.cpu() # Move to CPU
   output_str = tokenizer.decode(output[0], skip_special_tokens=True)
```

```eval_rst
.. note::

   The initial generation of optimized LLMs on Intel GPUs could be slow. Therefore, it's recommended to perform a **warm-up** run before the actual generation.
```


## QLoRA Finetuning

We also support finetuning LLMs (large language models) using Q-Lora with BigDL-LLM 4bit optimizations on Intel GPUs.

```eval_rst
.. note::

   Currently, only Hugging Face Transformers models are supported running Q-Lora finetuning.
```

To help you better understand the finetuning process, here we use model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example.

**Make sure you have prepared environment following instructions [here](../install_gpu.html). First of all, you need to import `intel_extension_for_pytorch` to run on Intel GPUs**:

```python
import intel_extension_for_pytorch as ipex
```

First, load model using `transformers`-style API and **set it to `to('xpu')`**. We specify `load_in_low_bit="nf4"` here to apply 4-bit NormalFloat optimization. According to the [QLoRA paper](https://arxiv.org/pdf/2305.14314.pdf), using `"nf4"` could yield better model quality than `"int4"`.

```python
import intel_extension_for_pytorch as ipex
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
