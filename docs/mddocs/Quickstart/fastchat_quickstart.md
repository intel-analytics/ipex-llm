# Serving using IPEX-LLM and FastChat

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

IPEX-LLM can be easily integrated into FastChat so that user can use `IPEX-LLM` as a serving backend in the deployment.

## Table of Contents
- [Install IPEX-LLM with FastChat](./fastchat_quickstart.md#1-install-ipex-llm-with-fastchat)
- [Start the Service](./fastchat_quickstart.md#2-start-the-service)


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
# [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

We have also provided an option `--load-low-bit-model` to load models that have been converted and saved into disk using the `save_low_bit` interface as introduced in this [document](../Overview/KeyFeatures/hugging_face_format.md#save--load).

Check the following examples:

```bash
# Or --device "cpu"
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path /Low/Bit/Model/Path --trust-remote-code --device "xpu" --load-low-bit-model
```

#### For self-speculative decoding example:

You can use IPEX-LLM to run `self-speculative decoding` example. Refer to [here](../../../python/llm/example/GPU/Speculative-Decoding) for more details on intel MAX GPUs. Refer to [here](../../../python/llm/example/CPU/Speculative-Decoding) for more details on intel CPUs.

```bash
# Available low_bit format only including bf16 on CPU.
source ipex-llm-init -t
python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "bf16" --trust-remote-code --device "cpu" --speculative

# Available low_bit format only including fp16 on GPU.
source /opt/intel/oneapi/setvars.sh
export ENABLE_SDP_FUSION=1
export SYCL_CACHE_PERSISTENT=1
# [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "fp16" --trust-remote-code --device "xpu" --speculative
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

We also provide the `vllm_worker` which uses the vLLM engine (on [CPU](../../../python/llm/example/CPU/vLLM-Serving) / [GPU](../../../python/llm/example/GPU/vLLM-Serving)) for better hardware utilization.

To run using the `vLLM_worker`, we don't need to change model name, just simply uses the following command:

```bash
# On CPU
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device cpu

# On GPU
source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
# [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python3 -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device xpu --load-in-low-bit "sym_int4" --enforce-eager
```

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).

#### Launch multiple workers

Sometimes we may want to start multiple workers for the best performance.  For running in CPU, you may want to seperate multiple workers in different sockets.  Assuming each socket have 48 physicall cores, then you may want to start two workers using the following example:

```bash
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "cpu" &

# All the workers other than the first worker need to specify a different worker port and corresponding worker-address
numactl -C 48-95 -m 1 python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "cpu" --port 21003 --worker-address "http://localhost:21003" &
```

For GPU, we may want to start two workers using different GPUs.  To achieve this, you should use `ZE_AFFINITY_MASK` environment variable to select different GPUs for different workers.  Below shows an example:

```bash
ZE_AFFINITY_MASK=1 python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "xpu" &

# All the workers other than the first worker need to specify a different worker port and corresponding worker-address
ZE_AFFINITY_MASK=2 python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --low-bit "sym_int4" --trust-remote-code --device "xpu" --port 21003 --worker-address "http://localhost:21003" &
```

If you are not sure the effect of `ZE_AFFINITY_MASK`, then you could set `ZE_AFFINITY_MASK` and check the result of `sycl-ls`.

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

### Launch TGI Style API server

When you have started the controller and the worker, you can start TGI Style API server as follows:

```bash
python3 -m ipex_llm.serving.fastchat.tgi_api_server --host localhost --port 8000
```
You can use `curl` for observing the output of the api

#### Using /generate API

This is to send a sentence as inputs in the request, and is expected to receive a response containing model-generated answer.

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "inputs": "What is AI?",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": true,
    "details": true,
    "do_sample": true,
    "frequency_penalty": 0.1,
    "grammar": {
      "type": "json",
      "value": "string"
    },
    "max_new_tokens": 32,
    "repetition_penalty": 1.03,
    "return_full_text": false,
    "seed": 0.1,
    "stop": [
      "photographer"
    ],
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "top_p": 0.95,
    "truncate": true,
    "typical_p": 0.95,
    "watermark": true
  }
}' http://localhost:8000/generate
```

Sample output:
```bash
{
    "details": {
        "best_of_sequences": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\nArtificial Intelligence (AI) is a branch of computer science that attempts to simulate the way that the human brain works. It is a branch of computer "
                },
                "finish_reason": "length",
                "generated_text": "\nArtificial Intelligence (AI) is a branch of computer science that attempts to simulate the way that the human brain works. It is a branch of computer ",
                "generated_tokens": 31
            }
        ]
    },
    "generated_text": "\nArtificial Intelligence (AI) is a branch of computer science that attempts to simulate the way that the human brain works. It is a branch of computer ",
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 35,
        "completion_tokens": 31
    }
}
```

#### Using /generate_stream API

This is to send a sentence as inputs in the request, and a long connection will be opened to continuously receive multiple responses containing model-generated answer.

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "inputs": "What is AI?",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": true,
    "details": true,
    "do_sample": true,
    "frequency_penalty": 0.1,
    "grammar": {
      "type": "json",
      "value": "string"
    },
    "max_new_tokens": 32,
    "repetition_penalty": 1.03,
    "return_full_text": false,
    "seed": 0.1,
    "stop": [
      "photographer"
    ],
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "top_p": 0.95,
    "truncate": true,
    "typical_p": 0.95,
    "watermark": true
  }
}' http://localhost:8000/generate_stream
```

Sample output:
```bash
data: {"token": {"id": 663359, "text": "", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 300560, "text": "\n", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 725120, "text": "Artificial Intelligence ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 734609, "text": "(AI) is ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 362235, "text": "a branch of computer ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 380983, "text": "science that attempts to ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 249979, "text": "simulate the way that ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 972663, "text": "the human brain ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 793301, "text": "works. It is a ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 501380, "text": "branch of computer ", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 673232, "text": "", "logprob": 0.0, "special": false}, "generated_text": null, "details": null, "special_ret": null}

data: {"token": {"id": 2, "text": "</s>", "logprob": 0.0, "special": true}, "generated_text": "\nArtificial Intelligence (AI) is a branch of computer science that attempts to simulate the way that the human brain works. It is a branch of computer ", "details": {"finish_reason": "eos_token", "generated_tokens": 31, "prefill_tokens": 4, "seed": 2023}, "special_ret": {"tensor": []}}
```


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
