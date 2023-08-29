# GPU Supports

You may apply INT4 optimizations to any Hugging Face *Transformers* models on device with Intel GPUs as follows:

```python
# import ipex
import intel_extension_for_pytorch as ipex

# load Hugging Face Transformers model with INT4 optimizations on Intel GPUs
from bigdl.llm.transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('/path/to/model/',
                                             load_in_4bit=True,
                                             optimize_model=False)
model = model.to('xpu')
```

```eval_rst
.. note::

   You may apply INT8 optimizations as follows:

   .. code-block:: python

      model = AutoModelForCausalLM.from_pretrained('/path/to/model/',
                                                   load_in_low_bit="sym_int8",
                                                   optimize_model=False)
      model = model.to('xpu')
```

After loading the Hugging Face Transformers model, you may easily run the optimized model as follows:

```python
# run the optimized model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...).to('xpu')
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids)
```

```eval_rst
.. seealso::

   See the complete examples `here <https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4/GPU>`_
```