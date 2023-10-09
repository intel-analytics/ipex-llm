## PyTorch API

In general, you just need one-line `optimize_model` to easily optimize any loaded PyTorch model, regardless of the library or API you are using. With BigDL-LLM, PyTorch models (in FP16/BF16/FP32) can be optimized with low-bit quantizations (supported precisions include INT4, INT5, INT8, etc).

First, use any PyTorch APIs you like to load your model. To help you better understand the process, here we use [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library `LlamaForCausalLM` to load a popular model [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as an example:

```python
# Create or load any Pytorch model, take Llama-2-7b-chat-hf as an example
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)
```

Then, just need to call `optimize_model` to optimize the loaded model and INT4 optimization is applied on model by default: 
```python
from bigdl.llm import optimize_model

# With only one line to enable BigDL-LLM INT4 optimization
model = optimize_model(model)
```

After optimizing the model, BigDL-LLM does not require any change in the inference code. You can use any libraries to run the optimized model with very low latency.

```eval_rst
.. seealso::

   * For more detailed usage of ``optimize_model``, please refer to the `API documentation <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/LLM/optimize.html>`_.
```
