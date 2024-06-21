# IPEX-LLM in 5 minutes

You can use IPEX-LLM to run any [*Hugging Face Transformers*](https://huggingface.co/docs/transformers/index) PyTorch model. It automatically optimizes and accelerates LLMs using low-precision (INT4/INT5/INT8) techniques, modern hardware accelerations and latest software optimizations.

Hugging Face transformers-based applications can run on IPEX-LLM with one-line code change, and you'll immediately observe significant speedup<sup><a href="#footnote-perf" id="ref-perf">[1]</a></sup>.

Here, let's take a relatively small LLM model, i.e [open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2), and IPEX-LLM INT4 optimizations as an example.

## Load a Pretrained Model

Simply use one-line `transformers`-style API in `ipex-llm` to load `open_llama_3b_v2` with INT4 optimization (by specifying `load_in_4bit=True`) as follows:

```python
from ipex_llm.transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="openlm-research/open_llama_3b_v2",
                                             load_in_4bit=True)
```

> [!TIP]
> [open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2) is a pretrained large language model hosted on Hugging Face. `openlm-research/open_llama_3b_v2` is its Hugging Face model id. `from_pretrained` will automatically download the model from Hugging Face to a local cache path (e.g. ``~/.cache/huggingface``), load the model, and converted it to `ipex-llm` INT4 format.
>
> It may take a long time to download the model using API. You can also download the model yourself, and set `pretrained_model_name_or_path` to the local path of the downloaded model. This way, `from_pretrained` will load and convert directly from local path without download.

## Load Tokenizer

You also need a tokenizer for inference. Just use the official `transformers` API to load `LlamaTokenizer`:

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path="openlm-research/open_llama_3b_v2")
```

## Run LLM

Now you can do model inference exactly the same way as using official `transformers` API:

```python
import torch

with torch.inference_mode():
    prompt = 'Q: What is CPU?\nA:'
    
    # tokenize the input prompt from string to token ids
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # predict the next tokens (maximum 32) based on the input token ids
    output = model.generate(input_ids,
                            max_new_tokens=32)

    # decode the predicted token ids to output string
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(output_str)
```

------

<div>
    <p>
        <sup><a href="#ref-perf" id="footnote-perf">[1]</a>
            Performance varies by use, configuration and other factors. <code><span>ipex-llm</span></code> may not optimize to the same degree for non-Intel products. Learn more at <a href="https://www.Intel.com/PerformanceIndex">www.Intel.com/PerformanceIndex</a>.
        </sup>
    </p>
</div>
