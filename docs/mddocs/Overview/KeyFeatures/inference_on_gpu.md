# Inference on GPU

Apart from the significant acceleration capabilites on Intel CPUs, IPEX-LLM also supports optimizations and acceleration for running LLMs (large language models) on Intel GPUs. With IPEX-LLM, PyTorch models (in FP16/BF16/FP32) can be optimized with low-bit quantizations (supported precisions include INT4, INT5, INT8, etc).

Compared with running on Intel CPUs, some additional operations are required on Intel GPUs. To help you better understand the process, here we use a popular model [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as an example.

**Make sure you have prepared environment following instructions [here](../install_gpu.md).**

> [!NOTE]
> If you are using an older version of `ipex-llm` (specifically, older than 2.5.0b20240104), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code. 

## Load and Optimize Model

You could choose to use [PyTorch API](./optimize_model.md) or [`transformers`-style API](./transformers_style_api.md) on Intel GPUs according to your preference.

**Once you have the model with IPEX-LLM low bit optimization, set it to `to('xpu')`**.

- For **PyTorch API**:

  You could optimize any PyTorch model with "one-line code change", and the loading and optimizing process on Intel GPUs maybe as follows:

  ```python
  # Take Llama-2-7b-chat-hf as an example
  from transformers import LlamaForCausalLM
  from ipex_llm import optimize_model

  model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)
  model = optimize_model(model) # With only one line to enable IPEX-LLM INT4 optimization

  model = model.to('xpu') # Important after obtaining the optimized model
  ```

  > **Tip**:
  >
  > When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the `optimize_model` function. This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
  >
  > See the [API doc](../../PythonAPI/optimize.md) for ``optimize_model`` to find more information.

  Especially, if you have saved the optimized model following setps [here](./optimize_model.md#save), the loading process on Intel GPUs maybe as follows:

  ```python
  from transformers import LlamaForCausalLM
  from ipex_llm.optimize import low_memory_init, load_low_bit

  saved_dir='./llama-2-ipex-llm-4-bit'
  with low_memory_init(): # Fast and low cost by loading model on meta device
    model = LlamaForCausalLM.from_pretrained(saved_dir,
                                              torch_dtype="auto",
                                              trust_remote_code=True)
  model = load_low_bit(model, saved_dir) # Load the optimized model

  model = model.to('xpu') # Important after obtaining the optimized model
  ```

- For **``transformers``-style API**:

  You could run any Hugging Face Transformers model with `transformers`-style API, and the loading and optimizing process on Intel GPUs maybe as follows:
  
  ```python
  # Take Llama-2-7b-chat-hf as an example
  from ipex_llm.transformers import AutoModelForCausalLM

  # Load model in 4 bit, which convert the relevant layers in the model into INT4 format
  model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', load_in_4bit=True)

  model = model.to('xpu') # Important after obtaining the optimized model
  ```

  > **Tip**:
  >
  > When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the `from_pretrained` function. This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
  >
  > See the [API doc](../../PythonAPI/transformers.md) to find more information.

  Especially, if you have saved the optimized model following setps [here](./hugging_face_format.md#save--load), the loading process on Intel GPUs maybe as follows:

  ```python
  from ipex_llm.transformers import AutoModelForCausalLM

  saved_dir='./llama-2-ipex-llm-4-bit'
  model = AutoModelForCausalLM.load_low_bit(saved_dir) # Load the optimized model

  model = model.to('xpu') # Important after obtaining the optimized model
  ```

  > **Tip**:
  >
  > When running saved optimized models on Intel iGPUs for Windows users, we also recommend setting `cpu_embedding=True` in the `load_low_bit` function.

## Run Optimized Model

You could then do inference using the optimized model on Intel GPUs almostly the same as on CPUs. **The only difference is to set `to('xpu')` for input tensors.**

Continuing with the [example of Llama-2-7b-chat-hf](#load-and-optimize-model), running as follows:
```python
import torch

with torch.inference_mode():
   prompt = 'Q: What is CPU?\nA:'
   input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu') # With .to('xpu') specifically for inference on Intel GPUs
   output = model.generate(input_ids, max_new_tokens=32)
   output_str = tokenizer.decode(output[0], skip_special_tokens=True)
```

> [!NOTE]
> The initial generation of optimized LLMs on Intel GPUs could be slow. Therefore, it's recommended to perform a **warm-up** run before the actual generation.

> [!NOTE]
> If you are a Windows user, please also note that for **the first time** that **each model** runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

> [!TIP]
> See the complete examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU).