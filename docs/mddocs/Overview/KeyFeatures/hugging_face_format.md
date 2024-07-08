# Hugging Face ``transformers`` Format

## Load in Low Precision
You may apply INT4 optimizations to any Hugging Face *Transformers* models as follows:

```python
# load Hugging Face Transformers model with INT4 optimizations
from ipex_llm.transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)
```

After loading the Hugging Face *Transformers* model, you may easily run the optimized model as follows:

```python
# run the optimized model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str, ...)
output_ids = model.generate(input_ids, ...)
output = tokenizer.batch_decode(output_ids)
```

> [!TIP]
> See the complete CPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels) and GPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HuggingFace).

> [!NOTE]
> You may apply more low bit optimizations (including INT8, INT5 and INT4) as follows:
>
> ```python
> model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_low_bit="sym_int5")
> ```
>
> See the CPU example [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types) and GPU example [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HuggingFace/More-Data-Types).


## Save & Load
After the model is optimized using INT4 (or INT8/INT5), you may save and load the optimized model as follows:

```python
model.save_low_bit(model_path)

new_model = AutoModelForCausalLM.load_low_bit(model_path)
```

> [!TIP]
> See the complete CPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load) and GPU examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HuggingFace/Save-Load).
