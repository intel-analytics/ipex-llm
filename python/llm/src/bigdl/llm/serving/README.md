## Serving using BigDL-LLM and FastChat

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

BigDL-LLM can be easily integrated into FastChat so that user can use `BigDL-LLM` as a serving backend in the deployment.

### Working with BigDL-LLM Serving

<details><summary>Table of Contents</summary>

- [Install](#install)
- [Models](#models)
- [Boot Service](#start-the-service)
  - [Web GUI](#serving-with-webgui)
  - [RESTful API](#serving-with-openai-compatible-restful-apis)
</details>

#### Install

You may install **`bigdl-llm`** with `FastChat` as follows:

```bash
pip install --pre --upgrade bigdl-llm[serving]

# Or
pip install --pre --upgrade bigdl-llm[all]
```

To add GPU support for FastChat, you may install **`bigdl-llm`** as follows:
```bash
pip install --pre --upgrade bigdl-llm[xpu, serving] -f https://developer.intel.com/ipex-whl-stable-xpu
```

#### Models

Using BigDL-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

FastChat determines the Model adapter to use through path matching. Therefore, in order to load models using BigDL-LLM, you need to make some modifications to the model's name.

For instance, assuming you have downloaded the `llama-7b-hf` from [HuggingFace](https://huggingface.co/decapoda-research/llama-7b-hf).  Then, to use the `BigDL-LLM` as backend, you need to change name from `llama-7b-hf` to `bigdl-7b`.
The key point here is that the model's path should include "bigdl" and **should not include paths matched by other model adapters**.

> note: This is caused by the priority of name matching list. The new added `BigDL-LLM` adapter is at the tail of the name-matching list so that it has the lowest priority. If model path contains other keywords like `vicuna` which matches to another adapter with higher priority, then the `BigDL-LLM` adapter will not work.

A special case is `ChatGLM` models. For these models, you do not need to do any changes after downloading the model and the `BigDL-LLM` backend will be used automatically.


#### Start the service

##### Serving with WebGUI

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

###### Launch the Controller
```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

###### Launch the model worker(s)
```bash
python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

> To run model worker using Intel GPU, simple change the --device cpu option to --device xpu

We also provide the `vllm_worker` which uses the [vLLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/vLLM-Serving) engine for better hardware utilization.

To run using the `vllm_worker`, just simply uses the following command:
```bash
python3 -m bigdl.llm.serving.vllm_worker --model-path meta-llama/Llama-2-7b-chat-hf --device cpu/xpu # based on your device
```

###### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with `BigDL-LLM` as the backend. You can open your browser and chat with a model now.

##### Serving with OpenAI-Compatible RESTful APIs

To start an OpenAI API server that provides compatible APIs using `BigDL-LLM` backend, you need three main components: an OpenAI API Server that serves the in-coming requests, model workers that host one or more models, and a controller to coordinate the web server and model workers.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s):

```bash
python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```