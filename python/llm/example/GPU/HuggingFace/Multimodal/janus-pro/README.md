# Janus-Pro
In this directory, you will find examples on how you could apply IPEX-LLM low-bit optimizations on Janus-Pro model on [Intel GPUs](../../../README.md). For illustration purposes, we utilize [deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B) and [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) as reference Janus-Pro models.

In the following examples, we will guide you to apply IPEX-LLM optimizations on Janus-Pro models for text/image inputs.

## 0. Requirements & Installation

To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 0.1 Install IPEX-LLM

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)** on Windows:
  ```cmd
  conda create -n llm python=3.11 libuv
  conda activate llm

  :: or --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/cn/
  pip install --pre --upgrade ipex-llm[xpu_lnl] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/us/
  ``` 
- For **Intel Arc B-Series GPU (code name Battlemage)** on Linux:
  ```cmd
  conda create -n llm python=3.11
  conda activate llm

  # or --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ``` 

> [!NOTE]
> We will update for installation on more Intel GPU platforms.

###  0.2 Install Required Pacakges for Janus-Pro

First, you need to clone `deepseek-ai/Janus` from GitHub.

```bash
git clone https://github.com/deepseek-ai/Janus.git
```

Then you can install the requirements for Janus-Pro models.

```bash
conda activate llm
cd Janus

# refer to https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
pip install -e .

pip install transformers==4.45.0
pip install accelerate==0.33.0
pip install "trl<0.12.0"

cd ..
```

### 0.3 Runtime Configuration

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)** on Windows:
  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  ``` 
- For **Intel Arc B-Series GPU (code name Battlemage)** on Linux:
  ```bash
  unset OCL_ICD_VENDOR
  export SYCL_CACHE_PERSISTENT=1
  ``` 

> [!NOTE]
> We will update for runtime configuration on more Intel GPU platforms.

## 1. Example: Predict Tokens using `generate()` API
In [generate.py](./generate.py), we show a use case for a Janus-Pro model to predict the next N tokens using `generate()` API based on text/image inputs, or a combination of two of them, with IPEX-LLM low-bit optimizations on Intel GPUs.

### 1.1 Running example

- Generate with text input
  - [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
    ```bash
    python generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --low-bit "sym_int8" --prompt PROMPT --n-predict N_PREDICT
    ```
  - [deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
    ```bash
    python generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --low-bit "sym_int8" --prompt PROMPT --n-predict N_PREDICT
    ```

- Generate with text + image inputs
  - [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
    ```bash
    python generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --low-bit "sym_int8" --prompt PROMPT --image-path IMAGE_PATH --n-predict N_PREDICT
    ```
  - [deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
    ```bash
    python generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --low-bit "sym_int8" --prompt PROMPT --image-path IMAGE_PATH --n-predict N_PREDICT
    ```

> [!NOTE]
> We recommand IPEX-LLM INT8 (`sym_int8`) optimizations for `deepseek-ai/Janus-Pro-1B` and `deepseek-ai/Janus-Pro-7B` to enhance output quality.

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for Janus-Pro model (e.g. `deepseek-ai/Janus-Pro-7B` or `deepseek-ai/Janus-Pro-1B`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'deepseek-ai/Janus-Pro-7B'`.
- `--prompt PROMPT`: argument defining the text input. It is default to be `'Describe the image in detail.'` when `--image-path` is provided. Otherwise, it is default to be `'What is AI?'`.
- `--image-path IMAGE_PATH`: argument defining the image input.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model.

### 1.2 Sample Outputs
The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a><br>
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg


#### [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)


- Chat with text + image inputs
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  5602445367_3504763978_z.jpg
  -------------------- Input Prompt (Formatted) --------------------
  You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

  <|User|>: <image_placeholder>
  Describe the image in detail.

  <|Assistant|>:
  -------------------- Chat Output --------------------
  The image shows a young child holding a small plush toy. The child is wearing a pink and white striped dress with a red and white bow on the shoulder.
  ```

- Chat with only text input:
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  None
  -------------------- Input Prompt (Formatted) --------------------
  You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

  <|User|>: What is AI?

  <|Assistant|>:
  -------------------- Chat Output --------------------
  AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think, learn, and make decisions like humans.
  ```

#### [deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)


- Chat with text + image inputs
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  5602445367_3504763978_z.jpg
  -------------------- Input Prompt (Formatted) --------------------
  You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

  <|User|>: <image_placeholder>
  Describe the image in detail.

  <|Assistant|>:
  -------------------- Chat Output --------------------
  The image shows a young child holding a small plush teddy bear. The teddy bear is dressed in a pink outfit with a polka-dotted tutu

  ```

- Chat with only text input:
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  None
  -------------------- Input Prompt (Formatted) --------------------
  You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

  <|User|>: What is AI?

  <|Assistant|>:
  -------------------- Chat Output --------------------
  AI stands for Artificial Intelligence. It is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence, such as learning
  ```
