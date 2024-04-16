# Serving using IPEX-LLM and FastChat

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

IPEX-LLM can be easily integrated into FastChat so that user can use `IPEX-LLM` as a serving backend in the deployment.

## Quick Start

This quickstart guide walks you through installing and running `FastChat` with `ipex-llm`.

## 1. Install IPEX-LLM with FastChat

To run on CPU, you can install ipex-llm as follows:

```bash
pip install --pre --upgrade ipex-llm[serving,all]
```

To add GPU support for FastChat, you may install **`ipex-llm`** as follows:

```bash
pip install --pre --upgrade ipex-llm[xpu,serving] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

```

## 2. Start the service

### Launch controller

You need first run the fastchat controller

```bash
python3 -m fastchat.serve.controller
```

If the controller run successfully, you can see the output like this:

```bash
Uvicorn running on http://localhost:21001
```

### Launch model worker(s) and load models

Using IPEX-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

#### IPEX-LLM worker

To integrate IPEX-LLM with `FastChat` efficiently, we have provided a new model_worker implementation named `ipex_llm_worker.py`.

```bash
# On CPU
# Available low_bit format including sym_int4, sym_int8, bf16 etc.
python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "cpu"

# On GPU
# Available low_bit format including sym_int4, sym_int8, fp16 etc.
source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

You can get output like this:

```bash
2024-04-12 18:18:09 | INFO | ipex_llm.transformers.utils | Converting the current model to sym_int4 format......
2024-04-12 18:18:11 | INFO | model_worker | Register to controller
2024-04-12 18:18:11 | ERROR | stderr | INFO:     Started server process [126133]
2024-04-12 18:18:11 | ERROR | stderr | INFO:     Waiting for application startup.
2024-04-12 18:18:11 | ERROR | stderr | INFO:     Application startup complete.
2024-04-12 18:18:11 | ERROR | stderr | INFO:     Uvicorn running on http://localhost:21002
```

For a full list of accepted arguments, you can refer to the main method of the `ipex_llm_worker.py`

#### IPEX-LLM vLLM worker

We also provide the `vllm_worker` which uses the [vLLM](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving) engine for better hardware utilization.

To run using the `vLLM_worker`,  we don't need to change model name, just simply uses the following command:

```bash
# On CPU
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device cpu

# On GPU
source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device xpu
```

### Launch Gradio web server

When you have started the controller and the worker, you can start web server as follows:

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/fastchat_gradio_web_ui.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/fastchat_gradio_web_ui.png" width=100%; />
</a>

By following these steps, you will be able to serve your models using the web UI with IPEX-LLM as the backend. You can open your browser and chat with a model now.

### Launch RESTful API server

To start an OpenAI API server that provides compatible APIs using IPEX-LLM backend, you can launch the `openai_api_server` and follow this [doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) to use it.

When you have started the controller and the worker, you can start RESTful API server as follows:

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

You can use `curl` for observing the output of the api

You can format the output using `jq`

#### List Models

```bash
curl http://localhost:8000/v1/models | jq
```

Example output

```json

{
  "object": "list",
  "data": [
    {
      "id": "Llama-2-7b-chat-hf",
      "object": "model",
      "created": 1712919071,
      "owned_by": "fastchat",
      "root": "Llama-2-7b-chat-hf",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-XpFyEE7Sewx4XYbEcdbCVz",
          "object": "model_permission",
          "created": 1712919071,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": true,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

#### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }' | jq
```

Example output

```json
{
  "id": "chatcmpl-jJ9vKSGkcDMTxKfLxK7q2x",
  "object": "chat.completion",
  "created": 1712919092,
  "model": "Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " Hello! My name is LLaMA, I'm a large language model trained by a team of researcher at Meta AI. Unterscheidung. 😊"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 53,
    "completion_tokens": 38
  }
}

```

#### Text Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-2-7b-chat-hf",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }' | jq
```

Example Output:

```json
{
  "id": "cmpl-PsAkpTWMmBLzWCTtM4r97Y",
  "object": "text_completion",
  "created": 1712919307,
  "model": "Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "text": ", in a far-off land, there was a magical kingdom called \"Happily Ever Laughter.\" It was a place where laughter was the key to happiness, and everyone who ",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 45,
    "completion_tokens": 40
  }
}

```
