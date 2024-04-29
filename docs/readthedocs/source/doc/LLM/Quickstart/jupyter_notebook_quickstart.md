# Use Jupyter Notebook to run model inference examples

This guide demonstrates how to use Jupyter Notebook to run model inference examples. 

It applies to MTL/WIN platform. Currently, ARC/LINUX platform is not supported.

# 0. Install Prerequisites
To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation on Windows.

visit the [Install IPEX-LLM on Windows with Intel GPU Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html), and follow [Install Prerequisites](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-prerequisites) to install Visual Studio 2022, GPU driver, Conda, and Intel® oneAPI Base Toolkit 2024.0. Then follow [Install ipex-llm](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-ipex-llm) to install ipex-llm for GPU.

# 1. Use Jupyter Notebook to run model inference examples

## 1.1 Use VS Code

### Install VS Code and dependency
- Download and install [VS Code for Windows](https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user).
- Install **Python** plugin.
<a href="https://llm-assets.readthedocs.io/en/latest/_images/python_extension.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/python_extension.png" width=100%; />
</a>

- Create and edit Notebook: Click `File`->`New File...`->`Jupyter Notebook` to create a new Notebook(e.g., llama2_generate.ipynb). Click `+ Code` to create a code cell and click the triangle on the left to execute the code. The output will be displayed directly below the code cell. Click `+ Markdown` to create a Markdown cell.
<a href="https://llm-assets.readthedocs.io/en/latest/_images/example_for_notebook.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/example_for_notebook.png" width=100%; />
</a>

- Select Kernel: Click `Select Kernel`->`Python Environments` and select the conda virtual environment we want to use.
<a href="https://llm-assets.readthedocs.io/en/latest/_images/select_kernel.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/select_kernel.png" width=100%; />
</a>

### Runtime Configurations
For optimal performance, it is recommended to set several environment variables. For Intel iGPU on windows, Create a new code cell in a Notebook and execute the cell:
```bash
import os
os.environ['SYCL_CACHE_PERSISTENT'] = '1'
os.environ['BIGDL_LLM_XMX_DISABLED'] = '1'
```
<a href="https://llm-assets.readthedocs.io/en/latest/_images/set_environ.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/set_environ.png" width=100%; />
</a>

>Note: If we encounter the following issue, just click `install`.
<a href="https://llm-assets.readthedocs.io/en/latest/_images/install_ipykernel.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/install_ipykernel.png" width=100%; />
</a>

### Load a pretrained Model and Tokenizer
Before using a LLM, we need to first load one. Here we take Llama2 as example. In this example, we use `ipex_llm.transformers.AutoModelForCausalLM` to load the `Llama-2-7b-chat-hf`. To enable INT4 optimization, simply set `load_in_4bit=True` in `from_pretrained`.

Create a new code cell in a Notebook and execute the cell:
```bash
import torch
import time
import argparse

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

model_path = "C:\\Users\Administrator\\llm-models\\Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            load_in_4bit=True,
                                            optimize_model=True,
                                            trust_remote_code=True,
                                            use_cache=True)
model = model.half().to('xpu')

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
```
<a href="https://llm-assets.readthedocs.io/en/latest/_images/load_model_and_tokenizer.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/load_model_and_tokenizer.png" width=100%; />
</a>

### Start inference
Now that the model is successfully loaded, we can start inference. 

Firstly we set the prompt in a new code cell and execute the cell:
```bash
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(message: str, chat_history: list[tuple[str, str]],
            system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

prompt = "What is AI?"
```

<a href="https://llm-assets.readthedocs.io/en/latest/_images/set_prompt.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/set_prompt.png" width=100%; />
</a>

Then We shall use the `Huggingface transformers` inference API to do this job. `max_new_tokens` parameter in the generate function defines the maximum number of tokens to predict.
```bash
with torch.inference_mode():
    prompt = get_prompt(prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
    # ipex_llm model needs a warmup, then inference time can be accurate
    output = model.generate(input_ids,
                            max_new_tokens=32)

    # start inference
    st = time.time()
    # if your selected model is capable of utilizing previous key/value attentions
    # to enhance decoding speed, but has `"use_cache": false` in its model config,
    # it is important to set `use_cache=True` explicitly in the `generate` function
    # to obtain optimal performance with IPEX-LLM INT4 optimizations
    output = model.generate(input_ids,
                            max_new_tokens=32)
    torch.xpu.synchronize()
    end = time.time()
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'Inference time: {end-st} s')
    print('-'*20, 'Prompt', '-'*20)
    print(prompt)
    print('-'*20, 'Output', '-'*20)
    print(output_str)
```

<a href="https://llm-assets.readthedocs.io/en/latest/_images/inference.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/inference.png" width=100%; />
</a>

## 1.2 Use browser

### Install and launch Jupyter Notebook
Open PowerShell, enter:
```bash
pip install Notebook
jupyter notebook --ip=0.0.0.0 --port=8888
```
This should launch Jupyter Notebook and open a new tab in our default web browser.
<a href="https://llm-assets.readthedocs.io/en/latest/_images/launch_notebook.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/launch_notebook.png" width=100%; />
</a>
<a href="https://llm-assets.readthedocs.io/en/latest/_images/notebook_browser.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/notebook_browser.png" width=100%; />
</a>

### Runtime Configurations
Open the notebook(e.g., llama2_generate.ipynb) in the browser and execute the cell as **Runtime Configurations** of VS Code.
</a>
<a href="https://llm-assets.readthedocs.io/en/latest/_images/configurations_for_browser.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/configurations_for_browser.png" width=100%; />
</a>   

### Load a pretrained Model and Tokenizer
Continue execute the cell as **Load a pretrained Model and Tokenizer** of VS Code.
</a>
<a href="https://llm-assets.readthedocs.io/en/latest/_images/load_model_for_browser.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/load_model_for_browser.png" width=100%; />
</a>

### Start inference
Continue execute the cell as **Start inference** of VS Code.
</a>
<a href="https://llm-assets.readthedocs.io/en/latest/_images/prompt_for_browser.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/prompt_for_browser.png" width=100%; />
</a>
</a>
<a href="https://llm-assets.readthedocs.io/en/latest/_images/inference_browser.png">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/inference_browser.png" width=100%; />
</a>