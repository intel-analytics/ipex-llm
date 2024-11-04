# Benchmark tool for transformers int4 (separate 1st token and rest)

[benchmark_util.py](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/utils/benchmark_util.py) is used to provide a simple benchmark tool for transformer int4 model to calculate 1st token performance and the rest on CPU and GPU.

## CPU Usage
Just put this file into your benchmark directory, and then wrap your transformer int4 model with `BenchmarkWrapper` (`model = BenchmarkWrapper(model)`).
Take `chatglm-6b` as an example:
```python
import torch
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer
from ipex_llm.utils import BenchmarkWrapper

model_path ='THUDM/chatglm-6b'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
model = BenchmarkWrapper(model, do_print=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "今天睡不着怎么办"
 
with torch.inference_mode():
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
```
Output will be like:
```bash
=========First token cost xx.xxxxs=========
=========Last token cost average xx.xxxxs (31 tokens in all)=========
```

## GPU Usage
### Inference on single GPU
Just put this file into your benchmark directory, and then wrap your transformer int4 model with `BenchmarkWrapper` (`model = BenchmarkWrapper(model)`).
Take `chatglm-6b` as an example:
```python
import torch
import intel_extension_for_pytorch as ipex
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer
from ipex_llm.utils import BenchmarkWrapper

model_path ='THUDM/chatglm-6b'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
model = model.to('xpu')
model = BenchmarkWrapper(model, do_print=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "今天睡不着怎么办"
 
with torch.inference_mode():
    # wamup two times as use ipex
    for i in range(2):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    # collect performance data now
    for i in range(5):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
```

### Sample Output
```bash
=========First token cost xx.xxxxs and 3.595703125 GB=========
=========Rest tokens cost average xx.xxxxs (31 tokens in all) and 3.595703125 GB=========
```

You can also set `verbose = True`
```python
model = BenchmarkWrapper(model, do_print=True, verbose=True)
```

```bash
=========First token cost xx.xxxxs and 3.595703125 GB=========
=========Rest token cost average xx.xxxxs (31 tokens in all) and 3.595703125 GB=========
Peak memory for every token: [3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125, 3.595703125]
```

### Inference on multi GPUs
Similarly, put this file into your benchmark directory, and then wrap your optimized model with `BenchmarkWrapper` (`model = BenchmarkWrapper(model)`).
For example, just need to apply following code patch on [Deepspeed Autotp example code](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/Deepspeed-AutoTP/deepspeed_autotp.py) to calculate 1st and the rest token performance:
```python
 import torch
 import transformers
 import deepspeed
 from ipex_llm.utils import BenchmarkWrapper
 
 def get_int_from_env(env_keys, default):
     """Returns the first positive env value found in the `env_keys` list or the default."""
@@ -98,6 +99,7 @@ if __name__ == '__main__':
     init_distributed()
 
     print(model)
+    model = BenchmarkWrapper(model, do_print=True)
 
     # Load tokenizer
     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```
