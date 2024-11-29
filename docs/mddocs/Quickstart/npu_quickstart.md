# Install IPEX-LLM on Windows with Intel NPU

IPEX-LLM provides NPU support for LLM acceleration on Intel Core™ Ultra Processers (Series 2), and offers both Python and C++ API. This guide not only demonstrates how to install IPEX-LLM on Windows with Intel NPU, but also includes a quick example of running an LLM on Intel NPU with IPEX-LLM Python/C++ API.

## Table of Contents

- [Install Prerequisites](#install-prerequisites)
- [Install `ipex-llm` with NPU Support](#install-ipex-llm-with-npu-support)
- [Runtime Configurations](#runtime-configurations)
- [A Quick Example](#a-quick-example)
  - [Python API](#python-api)
  - [C++ API](#c-api)
- [Accuracy Tuning](#accuracy-tuning)

## Install Prerequisites

> [!NOTE]
> IPEX-LLM NPU support on Windows has been verified on Intel Core™ Ultra Processers (Series 2) with processor number 2xxV (code name Lunar Lake).

### Update NPU Driver

> [!IMPORTANT]
> If you have NPU driver version lower than `31.0.100.3104`, it is highly recommended to update your NPU driver to the latest.

To update driver for Intel NPU:

1. Download the latest NPU driver

   - Visit the [official Intel NPU driver page for Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) and download the latest driver zip file.
   - Extract the driver zip file

2. Install the driver

   - Open **Device Manager** and locate **Neural processors** -> **Intel(R) AI Boost** in the device list
   - Right-click on **Intel(R) AI Boost** and select **Update driver**
   - Choose **Browse my computer for drivers**, navigate to the folder where you extracted the driver, and select **Next**
   - Wait for the installation finished

A system reboot is necessary to apply the changes after the installation is complete.

### (Optional) Install Visual Studio 2022

> [!NOTE]
> To use IPEX-LLM **C++ API** on Intel NPU, you are required to install Visual Studio 2022 on your system. If you plan to use the **Python API**, skip this step.

Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) Community Edition and select "Desktop development with C++" workload:

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_1.png"  width=80%/>
</div>

### Setup Python Environment

Visit [Miniforge installation page](https://conda-forge.org/download/), download the **Miniforge installer for Windows**, and follow the instructions to complete the installation.

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_miniforge_download.png"  width=80%/>
</div>

After installation, open the **Miniforge Prompt**, create a new python environment `llm-npu`:
```cmd
conda create -n llm python=3.11
```
Activate the newly created environment `llm-npu`:
```cmd
conda activate llm
```

> [!TIP]
> `ipex-llm` for NPU supports 3.10 and 3.11.

## Install `ipex-llm` with NPU Support

With the `llm-npu` environment active, use `pip` to install `ipex-llm` for NPU:

```cmd
conda create -n llm python=3.11 libuv
conda activate llm

pip install --pre --upgrade ipex-llm[npu]
```

## Runtime Configurations

For `ipex-llm` NPU support, set the following environment variable with active `llm-npu` environment:

```cmd
set BIGDL_USE_NPU=1
```

## A Quick Example

Now let's play with a real LLM on Intel NPU. We'll be using the [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model, a 7.61 billion parameter LLM for this demonstration. Follow the steps below to setup and run the model, and observe how it responds to a prompt "What is AI?". 

**IPEX-LLM on Intel NPU offers two API options: Python and C++**. You can choose the one that best suits your requirements.

### Python API

#### Step 1: Create `demo.py`

Create a new file named `demo.py` and insert the code snippet below to run [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model with IPEX-LLM optimizations on NPU:

```python
# Copy/Paste the contents to a new file demo.py

import torch
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer

print('Now start loading Tokenizer and optimizing Model...')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                          trust_remote_code=True)

# Load & optimize model using ipex-llm and load it to NPU
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
    load_in_low_bit="sym_int4",
    optimize_model=True,
    max_context_len=1024,
    max_prompt_len=512,
    mixed_precision=True,
    quantization_group_size=0,
    save_directory="./save_converted_model_dir"
)
print('Successfully loaded Tokenizer and optimized Model!')

# Format the prompt
# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#quickstart
question = "What is AI?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": question}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate predicted tokens
with torch.inference_mode():
    input_ids = tokenizer.encode(text, return_tensors="pt")

    output = model.generate(input_ids,
                            do_sample=False,
                            max_new_tokens=32)
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    print(output_str)
```

> [!TIP]
> When loading the model:
>
> - `ipex-llm` on NPU currently supports low-bit optimizations `load_in_low_bit='sym_int4'`/`'sym_int8'`.
> - `max_context_len` defines the maximum sequence length, which is the total number of token for both actual input and output combined.
> - The actual input token number should be smaller than `max_prompt_len`.


#### Step 2: Run `demo.py`

Run `demo.py` within the activated Python environment using the following command:

```cmd
python demo.py
```

#### Example output

TO BE ADDED

### C++ API

TO BE ADDED

#### Step 1: Convert model with `convert.py`
#### Step 2: Create `demo.cpp`
#### Step 3: Build `demo.exe`
#### Step 4: Run `demo.exe`
#### Example output

## Accuracy Tuning

IPEX-LLM provides several optimization methods for enhancing the accuracy of model outputs on Intel NPU. You can select and combine these techniques to achieve better outputs based on your specific use case.

### 1. Channel-Wise and Group-Wise Quantization

IPEX-LLM low-bit optimizations support both channel-wise and group-wise quantization on Intel NPU. When loading the model with Auto Model class from `ipex_llm.transformers.npu_model`, parameter `quantization_group_size` will control whether to use channel-wise or group-wise quantization.

If setting `quantization_group_size=0`, IPEX-LLM will use channel-wise quantization. If setting `quantization_group_size` larger than 0, e.g. `quantization_group_size=128`, IPEX-LLM will use group-wise quantization with group size to be 128.

You could try to use group-wise quantization for better outputs.

### 2. `IPEX_LLM_NPU_QUANTIZATION_OPT`

You could set environment variable `IPEX_LLM_NPU_QUANTIZATION_OPT=1` before loading the model to further enhance model accuracy of low-bit models.

### 3. Mixed Precision

When loading the model with Auto Model class from `ipex_llm.transformers.npu_model`, you could try to set parameter `mixed_precision=True` to enable mixed precision optimization when encountering output problems.