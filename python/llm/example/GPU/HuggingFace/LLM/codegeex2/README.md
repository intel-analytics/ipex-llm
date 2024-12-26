# CodeGeeX2

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on CodeGeeX2 models which is implemented based on the ChatGLM2 architecture trained on more code data on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b) (or [ZhipuAI/codegeex2-6b](https://www.modelscope.cn/models/ZhipuAI/codegeex2-6b) for ModelScope) as a reference CodeGeeX2 model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example 1: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a CodeGeeX2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.

### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# [optional] only needed if you would like to use ModelScope as model hub
pip install modelscope==1.11.0
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# [optional] only needed if you would like to use ModelScope as model hub
pip install modelscope==1.11.0
```

### 2. Download Model and Replace File
If you select the codegeex2-6b model ([THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b) (for **Hugging Face**) or [ZhipuAI/codegeex2-6b](https://www.modelscope.cn/models/ZhipuAI/codegeex2-6b) (for **ModelScope**)), please note that their code (`tokenization_chatglm.py`) initialized tokenizer after the call of `__init__` of its parent class, which may result in error during loading tokenizer. To address issue, we have provided an updated file ([tokenization_chatglm.py](./codegeex2-6b/tokenization_chatglm.py))

```python
def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
    self.tokenizer = SPTokenizer(vocab_file)
    super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
```

You could download the model from [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b) (for **Hugging Face**) or [ZhipuAI/codegeex2-6b](https://www.modelscope.cn/models/ZhipuAI/codegeex2-6b) (for **ModelScope**), and replace the file  `tokenization_chatglm.py` with [tokenization_chatglm.py](./codegeex2-6b/tokenization_chatglm.py).

### 3. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

### 5. Running examples
```bash
# for Hugging Face model hub
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT

# for ModelScope model hub
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --modelscope
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the **Hugging Face** (e.g. `THUDM/codegeex2-6b`) or **ModelScope** (e.g. `ZhipuAI/codegeex-6b`) repo id for the CodeGeeX2 model to be downloaded, or the path to the checkpoint folder. It is default to be `'THUDM/codegeex2-6b'` for **Hugging Face** or `'ZhipuAI/codegeex-6b'` for **ModelScope**.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'# language: Python\n# write a bubble sort function\n'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.
- `--modelscope`: using **ModelScope** as model hub instead of **Hugging Face**.

#### Sample Output
#### [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
# language: Python
# write a bubble sort function

-------------------- Output --------------------
# language: Python
# write a bubble sort function


def bubble_sort(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


print(bubble_sort([5, 2, 3, 4, 1]))
```
