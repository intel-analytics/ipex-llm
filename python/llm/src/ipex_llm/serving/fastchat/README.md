# Serving using IPEX-LLM and FastChat

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

IPEX-LLM can be easily integrated into FastChat so that user can use `IPEX-LLM` as a serving backend in the deployment.

<details>
<summary>Table of contents</summary>

- [Install](#install)
- [Start the service](#start-the-service)
  - [Launch controller](#launch-controller)
  - [Launch model worker(s) and load models](#launch-model-workers-and-load-models)
    - [IPEX-LLM model worker (deprecated)](#ipex-llm-model-worker-deprecated)
    - [IPEX-LLM worker](#ipex-llm-worker)
    - [IPEX-LLM vLLM worker](#ipex-llm-vllm-worker)
  - [Launch Gradio web server](#launch-gradio-web-server)
  - [Launch RESTful API server](#launch-restful-api-server)

</details>

## Install

You may install **`ipex-llm`** with `FastChat` as follows:

```bash
pip install --pre --upgrade ipex-llm[serving]
pip install transformers==4.36.0

# Or
pip install --pre --upgrade ipex-llm[all]

```

To add GPU support for FastChat, you may install **`ipex-llm`** as follows:

```bash
pip install --pre --upgrade ipex-llm[xpu,serving] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

```

## Start the service

### Launch controller

You need first run the fastchat controller

```bash
python -m fastchat.serve.controller
```

### Launch model worker(s) and load models

Using IPEX-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

#### IPEX-LLM worker

To integrate IPEX-LLM with `FastChat` efficiently, we have provided a new model_worker implementation named `ipex_llm_worker.py`.

To run the `ipex_llm_worker` on CPU, using the following code:

```bash
source ipex-llm-init -t

# Available low_bit format including sym_int4, sym_int8, bf16 etc.
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "cpu"
```

For GPU example:

```bash
# Available low_bit format including sym_int4, sym_int8, fp16 etc.
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

We have also provided an option `--load-low-bit-model` to load models that have been converted and saved into disk using the `save_low_bit` interface as introduced in this [document](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load/README.md).

Check the following examples:
```bash
# Or --device "cpu"
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path /Low/Bit/Model/Path --trust-remote-code --device "xpu" --load-low-bit-model
```

#### For self-speculative decoding example:

You can use IPEX-LLM to run `self-speculative decoding` example. Refer to [here](https://github.com/intel-analytics/ipex-llm/tree/c9fac8c26bf1e1e8f7376fa9a62b32951dd9e85d/python/llm/example/GPU/Speculative-Decoding) for more details on intel MAX GPUs. Refer to [here](https://github.com/intel-analytics/ipex-llm/tree/c9fac8c26bf1e1e8f7376fa9a62b32951dd9e85d/python/llm/example/GPU/Speculative-Decoding) for more details on intel CPUs.

```bash
# Available low_bit format only including bf16 on CPU.
source ipex-llm-init -t
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "bf16" --trust-remote-code --device "cpu" --speculative

# Available low_bit format only including fp16 on GPU.
source /opt/intel/oneapi/setvars.sh
export ENABLE_SDP_FUSION=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "fp16" --trust-remote-code --device "xpu" --speculative
```

For a full list of accepted arguments, you can refer to the main method of the `ipex_llm_worker.py`

#### IPEX-LLM vLLM worker

We also provide the `vllm_worker` which uses the [vLLM](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/vLLM-Serving) engine for better hardware utilization.

To run using the `vLLM_worker`,  we don't need to change model name, just simply uses the following command:

```bash
# On CPU
python -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device cpu

# On GPU
python -m ipex_llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device xpu
```

### Launch Gradio web server

```bash
python -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

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

```bash
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```
