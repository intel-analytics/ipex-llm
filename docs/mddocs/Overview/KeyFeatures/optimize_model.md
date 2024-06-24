## PyTorch API

In general, you just need one-line `optimize_model` to easily optimize any loaded PyTorch model, regardless of the library or API you are using. With IPEX-LLM, PyTorch models (in FP16/BF16/FP32) can be optimized with low-bit quantizations (supported precisions include INT4, INT5, INT8, etc).

### Optimize model

First, use any PyTorch APIs you like to load your model. To help you better understand the process, here we use [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library `LlamaForCausalLM` to load a popular model [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as an example:

```python
# Create or load any Pytorch model, take Llama-2-7b-chat-hf as an example
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)
```

Then, just need to call `optimize_model` to optimize the loaded model and INT4 optimization is applied on model by default: 
```python
from ipex_llm import optimize_model

# With only one line to enable IPEX-LLM INT4 optimization
model = optimize_model(model)
```

After optimizing the model, IPEX-LLM does not require any change in the inference code. You can use any libraries to run the optimized model with very low latency.

### More Precisions

In the [Optimize Model](#optimize-model), symmetric INT4 optimization is applied by default. You may apply other low bit optimizations (INT5, INT8, etc) by specifying the ``low_bit`` parameter.

Currently, ``low_bit`` supports options 'sym_int4', 'asym_int4', 'sym_int5', 'asym_int5' or 'sym_int8', in which 'sym' and 'asym' differentiate between symmetric and asymmetric quantization. Symmetric quantization allocates bits for positive and negative values equally, whereas asymmetric quantization allows different bit allocations for positive and negative values.

You may apply symmetric INT8 optimization as follows:

```python
from ipex_llm import optimize_model

# Apply symmetric INT8 optimization
model = optimize_model(model, low_bit="sym_int8")
```

### Save & Load Optimized Model

The loading process of the original model may be time-consuming and memory-intensive. For example, the [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model is stored with float16 precision, resulting in large memory usage when loaded using `LlamaForCausalLM`. To avoid high resource consumption and expedite loading process, you can use `save_low_bit` to store the model after low-bit optimization. Then, in subsequent uses, you can opt to use the `load_low_bit` API to directly load the optimized model. Besides, saving and loading operations are platform-independent, regardless of their operating systems.
#### Save

Continuing with the [example of Llama-2-7b-chat-hf](#optimize-model), we can save the previously optimized model as follows:
```python
saved_dir='./llama-2-ipex-llm-4-bit'
model.save_low_bit(saved_dir)
```
#### Load

We recommend to use the context manager `low_memory_init` to quickly initiate a model instance with low cost, and then use `load_low_bit` to load the optimized low-bit model as follows:
```python
from ipex_llm.optimize import low_memory_init, load_low_bit
with low_memory_init(): # Fast and low cost by loading model on meta device
   model = LlamaForCausalLM.from_pretrained(saved_dir,
                                            torch_dtype="auto",
                                            trust_remote_code=True)
model = load_low_bit(model, saved_dir) # Load the optimized model
```


> [!NOTE]
> - Please refer to the [API documentation](../../PythonAPI/optimize.md) for more details.
> - We also provide detailed examples on how to run PyTorch models (e.g., Openai Whisper, LLaMA2, ChatGLM2, Falcon, MPT, Baichuan2, etc.) using IPEX-LLM. See the complete CPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/PyTorch-Models) and GPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/PyTorch-Models)

