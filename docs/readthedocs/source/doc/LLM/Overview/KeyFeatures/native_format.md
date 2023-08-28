# `transformers`-style API: Native Format

You may also convert Hugging Face *Transformers* models into native INT4 format for maximum performance as follows.

```eval_rst
.. note::

   Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; you may use the corresponding API to load the converted model. (For other models, you can use the Hugging Face format as described `here <./hugging_face_format.html>`_).
```

```python
# convert the model
from bigdl.llm import llm_convert
bigdl_llm_path = llm_convert(model='/path/to/model/',
       outfile='/path/to/output/', outtype='int4', model_family="llama")

# load the converted model
# switch to ChatGLMForCausalLM/GptneoxForCausalLM/BloomForCausalLM/StarcoderForCausalLM to load other models
from bigdl.llm.transformers import LlamaForCausalLM
llm = LlamaForCausalLM.from_pretrained("/path/to/output/model.bin", native=True, ...)

# run the converted model
input_ids = llm.tokenize(prompt)
output_ids = llm.generate(input_ids, ...)
output = llm.batch_decode(output_ids)
```

```eval_rst
.. seealso::
   
   See the complete example `here <https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4/native_int4_pipeline.py>`_
```