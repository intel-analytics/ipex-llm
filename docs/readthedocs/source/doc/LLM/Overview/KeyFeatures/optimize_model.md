## General PyTorch Model Supports

You may apply BigDL-LLM optimizations on any Pytorch models, not only Hugging Face *Transformers* models for acceleration. With BigDL-LLM, PyTorch models (in FP16/BF16/FP32) can be optimized with low-bit quantizations (supported precisions include INT4/INT5/INT8).

You can easily enable BigDL-LLM INT4 optimizations on any Pytorch models just as follows:

```python
# Create or load any Pytorch model
model = ...

# Add only two lines to enable BigDL-LLM INT4 optimizations on model
from bigdl.llm import optimize_model
model = optimize_model(model)
```

After optimizing the model, you may straightly run the optimized model with no API changed and less inference latency.

```eval_rst
.. seealso::

   See the examples for Hugging Face *Transformers* models `here <https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/general_int4>`_. And examples for other general Pytorch models can be found `here <https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/pytorch-model>`_.
```
