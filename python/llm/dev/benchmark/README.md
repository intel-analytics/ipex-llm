# Benchmark tool for transformers int4 (separate 1st token and rest)

`benchmark_util.py` is used to provide a simple benchmark tool for transformer int4 model to calculate 1st token performance and the rest.

## Usage
Just put this file into your benchmark directory, and then wrap your transformer int4 model with `BenchmarkWrapper` (`model = BenchmarkWrapper(model)`).
Take `chatglm-6b` as an example:
```python
import torch
import os
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
import time
import numpy as np
from benchmark_util import BenchmarkWrapper

model_path ='THUDM/chatglm-6b'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
model = BenchmarkWrapper(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "今天睡不着怎么办"
 
with torch.inference_mode():
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
```
Output will be like:
```bash
=========First token cost 0.0518s=========
=========Last token cost average 0.0316s (31 tokens in all)=========
```
