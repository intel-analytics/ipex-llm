# Moonlight

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Moonlight model on [Intel GPUs](../../../README.md). For illustration purposes, we utilize [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) as reference Moonlight model.

## 0. Requirements & Installation

To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 0.1 Installation

```bash
conda create -n llm python=3.11
conda activate llm

# install IPEX-LLM with PyTorch 2.6 supports
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu

pip install transformers==4.45.0
pip install accelerate==0.33.0
pip install "trl<0.12.0" 

pip install tiktoken blobfile
```

### 0.2 Runtime Configuration

- For Windows users:
  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  :: optional
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ``` 

- For Linux users:
  ```cmd
  unset OCL_ICD_VENDOR
  export SYCL_CACHE_PERSISTENT=1
  # optional
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ``` 

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. Enabling this mode may improve performance, but sometimes this may also cause performance degradation. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)

## 1. Download & Convert Model

To run the Moonlight model with IPEX-LLM optimizations, we need to download and convert first it to make sure it could be successfully loaded by `transformers`.

### 1.1 Download Model

To download [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) from Hugging Face, you could use [download.py](./download.py) through:

```bash
download.py --repo-id moonshotai/Moonlight-16B-A3B-Instruct --commit-id 95583251e616c46a80715897a705cd38659afc27 
```

By default, Moonlight-16B-A3B-Instruct will be downloaded to the current folder. You could also define the download folder path by `--download-dir-path DOWNLOAD_DIR_PATH`.

> [!TIP]
> Refer to [here](https://huggingface.co/docs/hub/en/models-downloading) for althernative methods to download models from Hugging Face.
>
> For [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct), please make sure to use its revision/commit id `95583251e616c46a80715897a705cd38659afc27`.

### 1.2 Convert Model

Next, convert the downloaded model by [convert.py](./convert.py):

```bash
convert.py --model-path DOWNLOAD_DIR_PATH
```

The converted model will be saved at `<DOWNLOAD_DIR_PATH>-converted`.

## 2. Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a Moonlight model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.

### 2.1 Running example

```bash
python generate.py --converted-model-path `<DOWNLOAD_DIR_PATH>-converted` --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--converted-model-path CONVERTED_MODEL_PATH`: argument defining the converted model path by [`convert.py`](./convert.py)
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

### 2.2 Sample Outputs

#### [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
Is 123 a prime?
-------------------- Output --------------------
<|im_system|>system<|im_middle|>You are a helpful assistant provided by Moonshot-AI.<|im_end|><|im_user|>user<|im_middle|>Is 123 a prime?<|im_end|><|im_assistant|>assistant<|im_middle|>No, 123 is not a prime number. A prime number is a number greater than 1 that has no positive divisors other than 1 and itself
```
