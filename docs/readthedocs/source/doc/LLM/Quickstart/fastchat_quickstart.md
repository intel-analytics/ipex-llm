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
python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

For a full list of accepted arguments, you can refer to the main method of the `ipex_llm_worker.py`

#### IPEX-LLM vLLM worker

We also provide the `vllm_worker` which uses the [vLLM](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving) engine for better hardware utilization.

To run using the `vLLM_worker`,  we don't need to change model name, just simply uses the following command:

```bash
# On CPU
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device cpu

# On GPU
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device xpu
```

### Launch Gradio web server

When you have started the controller and the worker, you can start web server as follows:

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with IPEX-LLM as the backend. You can open your browser and chat with a model now.

### Launch RESTful API server

To start an OpenAI API server that provides compatible APIs using IPEX-LLM backend, you can launch the `openai_api_server` and follow this [doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) to use it.

When you have started the controller and the worker, you can start RESTful API server as follows:

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
