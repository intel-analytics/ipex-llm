# vLLM Serving with IPEX-LLM on Intel GPUs via Docker

This guide demonstrates how to run `vLLM` serving with `IPEX-LLM` on Intel GPUs via Docker.

## Install docker

Follow the instructions in this [guide](./docker_windows_gpu.md#linux) to install Docker on Linux.

## Pull the latest image

*Note: For running vLLM serving on Intel GPUs, you can currently use either the `intelanalytics/ipex-llm-serving-xpu:latest` or `intelanalytics/ipex-llm-serving-vllm-xpu:latest` Docker image.*

```bash
# This image will be updated every day
docker pull intelanalytics/ipex-llm-serving-xpu:latest
```

## Start Docker Container

 To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. Change the `/path/to/models` to mount the models.

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
export CONTAINER_NAME=ipex-llm-serving-xpu-container
sudo docker run -itd \
        --net=host \
        --group-add video \
        --device=/dev/dri \
        -v /path/to/models:/llm/models \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-serving-xpu-container /bin/bash
```

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2024.17.5.0.08_160000.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) w5-3435X OpenCL 3.0 (Build 0) [2024.17.5.0.08_160000.xmain-hotfix]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.35.27191.9]
[opencl:gpu:3] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.35.27191.9]
[opencl:gpu:4] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.35.27191.9]
[opencl:gpu:5] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.35.27191.9]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.27191]
[ext_oneapi_level_zero:gpu:1] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.27191]
[ext_oneapi_level_zero:gpu:2] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.27191]
[ext_oneapi_level_zero:gpu:3] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.27191]
```

## Running vLLM serving with IPEX-LLM on Intel GPU in Docker

We have included multiple vLLM-related files in `/llm/`:

1. `vllm_offline_inference.py`: Used for vLLM offline inference example,
    1. Modify following parameters in LLM class(line 48):

    |parameters|explanation|
    |:---|:---|
    |`model="YOUR_MODEL"`| the model path in docker, for example `"/llm/models/Llama-2-7b-chat-hf"`|
    |`load_in_low_bit="fp8"`| model quantization accuracy, acceptable ``'sym_int4'``, ``'asym_int4'``,  ``'fp6'``, ``'fp8'``, ``'fp8_e4m3'``, ``'fp8_e5m2'``,  ``'fp16'``; ``'sym_int4'`` means symmetric int 4, ``'asym_int4'`` means asymmetric int 4, etc. Relevant low bit optimizations will be applied to the model. default is ``'fp8'``, which is the same as ``'fp8_e5m2'``|
    |`tensor_parallel_size=1`| number of tensor parallel replicas, default is `1`|
    |`pipeline_parallel_size=1`| number of pipeline stages, default is `1`|

    2. Run the python script

    ```bash
    python vllm_offline_inference.py
    ```

    3. The expected output should be as follows:

```bash
INFO 09-25 21:37:31 gpu_executor.py:108] # GPU blocks: 747, # CPU blocks: 512
Processed prompts: 100%|█| 4/4 [00:22<00:00,  5.59s/it, est. speed input: 1.21 toks/s, output: 2.86 toks
Prompt: 'Hello, my name is', Generated text: ' [Your Name], and I am a member of the [Your Group Name].'
Prompt: 'The president of the United States is', Generated text: ' the head of the executive branch and the highest-ranking official in the federal'
Prompt: 'The capital of France is', Generated text: " Paris. It is the country's largest city and is known for its icon"
Prompt: 'The future of AI is', Generated text: ' vast and complex, with many different areas of research and application. Here are some'
  ```

2. `benchmark_vllm_throughput.py`: Used for benchmarking throughput
3. `payload-1024.lua`: Used for testing request per second using 1k-128 request
4. `start-vllm-service.sh`: Used for template for starting vLLM service

Before performing benchmark or starting the service, you can refer to this [section](../Quickstart/install_linux_gpu.md#runtime-configurations) to setup our recommended runtime configurations.

### Serving
>
> A script named `/llm/start-vllm-service.sh` have been included in the image for starting the service conveniently. You can tune the service using these four arguments:

|parameters|explanation|
|:---|:---|
|`--gpu-memory-utilization`| The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9.|
|`--max-model-len`| Model context length. If unspecified, will be automatically derived from the model config.|
|`--max-num-batched-token`| Maximum number of batched tokens per iteration.|
|`--max-num-seq`| Maximum number of sequences per iteration. Default: 256|
|`--block-size`| vLLM block size. Set to 8 to achieve a performance boost.|

#### Single card serving

Here are the steps to serve on a single card.

1. Modify the `model` and `served_model_name` in the script so that it fits your requirement. The `served_model_name` indicates the model name used in the API, for example:

```bash
model="/llm/models/Llama-2-7b-chat-hf"
served_model_name="llama2-7b-chat"
```

2. Start the service using `bash /llm/start-vllm-service.sh`, if the service have booted successfully, you should see the output similar to the following figure:
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" target="_blank">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" width=100%; />

  </a>
3. Using following curl command to test the server

```bash
curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "llama2-7b-chat",
          "prompt": "San Francisco is a",
          "max_tokens": 128
         }'
```

The expected output should be as follows:

```json
{
    "id": "cmpl-0a86629065c3414396358743d7823385",
    "object": "text_completion",
    "created": 1727273935,
    "model": "llama2-7b-chat",
    "choices": [
        {
            "index": 0,
            "text": "city that is known for its iconic landmarks, vibrant culture, and diverse neighborhoods. Here are some of the top things to do in San Francisco:. Visit Alcatraz Island: Take a ferry to the infamous former prison and experience the history of Alcatraz Island.2. Explore Golden Gate Park: This sprawling urban park is home to several museums, gardens, and the famous Japanese Tea Garden.3. Walk or Bike the Golden Gate Bridge: Take in the stunning views of the San Francisco Bay and the bridge from various v",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 133,
        "completion_tokens": 128
    }
}
```

#### Multi-card serving

For larger models (greater than 10b), we need to use multiple graphics cards for deployment. In the above script(`/llm/start-vllm-service.sh`), we need to make some modifications to achieve multi-card serving.

1. **Tensor Parallel Serving**: need modify the `-tensor-parallel-size` num, for example, using 2 cards for tp serving, add following parameter:

```bash
--tensor-parallel-size 2
```

or shortening:

```bash
-tp 2
```

2. **Pipeline Parallel Serving**: need modify the `-pipeline-parallel-size` num, for example, using 2 cards for pp serving, add following parameter:

```bash
--pipeline-parallel-size 2
```

or shortening:

```bash
-pp 2
```

3. **TP+PP Serving**: using tensor-parallel and pipline-parallel mixed, for example, if you have 4 GPUs in 2 nodes (2GPUs per node), you can set the tensor parallel size to 2 and the pipeline parallel size to 2.

```bash
--pipeline-parallel-size 2 \
--tensor-parallel-size 2
```

or shortening:

```bash
-pp 2 \
-tp 2
```

### Quantization

Quantizing model from FP16 to INT4 can effectively reduce the model size loaded into gpu memory by about 70 %. The main advantage is lower delay and memory usage.

#### IPEX-LLM

Two scripts are provided in the docker image for model inference.

1. vllm offline inference: `vllm_offline_inference.py`

> Only need change the `load_in_low_bit` value to use different quantization dtype. Commonly supported dtype containes:`sym_int4`, `fp6`, `fp8`, and `fp16`, full supported dtype refer to [load_in_low_bit](./vllm_docker_quickstart.md#running-vllm-serving-with-ipex-llm-on-intel-gpu-in-docker) in the llm class parameter table.

```python
llm = LLM(model="YOUR_MODEL",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          # Simply change here for the desired load_in_low_bit value
          load_in_low_bit="sym_int4",
          tensor_parallel_size=1,
          trust_remote_code=True)
```

then run

```bash
python vllm_offline_inference.py
```

2. vllm online service `start-vllm-service.sh`

> To fully utilize the continuous batching feature of the vLLM, you can send requests to the service using curl or other similar methods. The requests sent to the engine will be batched at token level. Queries will be executed in the same forward step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.
  
Modify the `--load-in-low-bit` value to `fp6`, `fp8`, `fp8_e4m3` or `fp16`

```bash
 # Change value --load-in-low-bit to [fp6, fp8, fp8_e4m3, fp16] to use different low-bit formats
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \ 
  --trust-remote-code \
  --block-size 8 \
  --gpu-memory-utilization 0.9 \
  --device xpu \ 
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4000 \
  --tensor-parallel-size 1 \
  --disable-async-output-proc \
  --distributed-executor-backend ray
```
  
then run following command to start vllm service

```bash
bash start-vllm-service.sh
```
  
Lastly, using curl command to send a request to service, below shows an example output using `Qwen1.5-7B-Chat` with low-bit format `sym_int4`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" width=100%; />
</a>

#### AWQ

Use AWQ as a way to reduce memory footprint. Firstly download the model after awq quantification, taking `Llama-2-7B-Chat-AWQ` as an example, download it on <https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ>

1. Offline inference usage with `/llm/vllm_offline_inference.py`

    1. Change the `/llm/vllm_offline_inference.py` LLM class code block's parameters `model`, `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

    ```python
    llm = LLM(model="/llm/models/Llama-2-7B-chat-AWQ/",
              quantization="AWQ",
              load_in_low_bit="asym_int4",
              device="xpu",
              dtype="float16",
              enforce_eager=True,
              tensor_parallel_size=1)
    ```

    then run the following command

    ```bash
    python vllm_offline_inference.py
    ```

    2. Expected result shows as below:

    ```bash
    2024-09-29 10:06:34,272 - INFO - Converting the current model to asym_int4 format......
    2024-09-29 10:06:34,272 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-09-29 10:06:40,080 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-09-29 10:06:41,258 - INFO - Loading model weights took 3.7381 GB
    WARNING 09-29 10:06:47 utils.py:564] Pin memory is not supported on XPU.
    INFO 09-29 10:06:47 gpu_executor.py:108] # GPU blocks: 1095, # CPU blocks: 512
    Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.67s/it, est. speed input: 1.19 toks/s, output: 2.82 toks/s]
    Prompt: 'Hello, my name is', Generated text: ' [Your Name], and I am a resident of [Your City/Town'
    Prompt: 'The president of the United States is', Generated text: ' the head of the executive branch and is one of the most powerful political figures in'
    Prompt: 'The capital of France is', Generated text: ' Paris. It is the most populous urban agglomeration in the European'
    Prompt: 'The future of AI is', Generated text: ' vast and exciting, with many potential applications across various industries. Here are'
    r
    ```

2. Online serving usage with `/llm/start-vllm-service.sh`
    1. Change the `/llm/start-vllm-service.sh`, set `model` parameter to awq model path and `served_model_name`. Add `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

    ```bash
    #!/bin/bash
    model="/llm/models/Llama-2-7B-Chat-AWQ/"
    served_model_name="llama2-7b-awq"
    ...
    python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
      --served-model-name $served_model_name \
      --model $model \
      ...
      --quantization awq \
      --load-in-low-bit asym_int4 \
      ...
    ```

    2. Use `bash start-vllm-service.sh` to start awq model online serving. Serving start successfully log:

    ```bash
    2024-10-18 01:50:24,124 - INFO - Converting the current model to asym_int4 format......
    2024-10-18 01:50:24,124 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-18 01:50:29,812 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-18 01:50:30,880 - INFO - Loading model weights took 3.7381 GB
    WARNING 10-18 01:50:39 utils.py:564] Pin memory is not supported on XPU.
    INFO 10-18 01:50:39 gpu_executor.py:108] # GPU blocks: 2254, # CPU blocks: 1024
    WARNING 10-18 01:50:39 serving_embedding.py:171] embedding_mode is False. Embedding API will not work.
    INFO 10-18 01:50:39 launcher.py:14] Available routes are:
    INFO 10-18 01:50:39 launcher.py:22] Route: /openapi.json, Methods: HEAD, GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /docs, Methods: HEAD, GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /docs/oauth2-redirect, Methods: HEAD, GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /redoc, Methods: HEAD, GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /health, Methods: GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /tokenize, Methods: POST
    INFO 10-18 01:50:39 launcher.py:22] Route: /detokenize, Methods: POST
    INFO 10-18 01:50:39 launcher.py:22] Route: /v1/models, Methods: GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /version, Methods: GET
    INFO 10-18 01:50:39 launcher.py:22] Route: /v1/chat/completions, Methods: POST
    INFO 10-18 01:50:39 launcher.py:22] Route: /v1/completions, Methods: POST
    INFO 10-18 01:50:39 launcher.py:22] Route: /v1/embeddings, Methods: POST
    INFO:     Started server process [995]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

    3. In docker send request to verfiy the serving status.

    ```bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "llama2-7b-awq",
              "prompt": "San Francisco is a",
              "max_tokens": 128
            }'
    ```

    and should get following output:

    ```json
    {
      "id": "cmpl-992e4c8463d24d0ab2e59e706123ef0d",
      "object": "text_completion",
      "created": 1729187735,
      "model": "llama2-7b-awq",
      "choices": [
        {
          "index": 0,
          "text": " food lover's paradise with a diverse array of culinary options to suit any taste and budget. Here are some of the top attractions when it comes to food and drink in San Francisco:\n\n1. Fisherman's Wharf: This bustling waterfront district is known for its fresh seafood, street performers, and souvenir shops. Be sure to try some of the local specialties like Dungeness crab, abalone, or sourdough bread.\n\n2. Chinatown: San Francisco's Chinatown is one of the largest and oldest",
          "logprobs": null,
          "finish_reason": "length",
          "stop_reason": null
        }
      ],
      "usage": {
        "prompt_tokens": 5,
        "total_tokens": 133,
        "completion_tokens": 128
      }
    }
    ```

#### GPTQ

Use GPTQ as a way to reduce memory footprint. Firstly download the model after gptq quantification, taking `Llama-2-13B-Chat-GPTQ` as an example, download it on <https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ>

1. Offline inference usage with `/llm/vllm_offline_inference.py`
    1. Change the `/llm/vllm_offline_inference` LLM class code block's parameters `model`, `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

    ```python
    llm = LLM(model="/llm/models/Llama-2-7B-Chat-GPTQ/",
              quantization="GPTQ",
              load_in_low_bit="asym_int4",
              device="xpu",
              dtype="float16",
              enforce_eager=True,
              tensor_parallel_size=1)
    ```

    then run the following command

    ```bash
    python vllm_offline_inference.py
    ```

    2. Expected result shows as below:

    ```bash
    2024-10-08 10:55:18,296 - INFO - Converting the current model to asym_int4 format......
    2024-10-08 10:55:18,296 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-08 10:55:23,478 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-08 10:55:24,581 - INFO - Loading model weights took 3.7381 GB
    WARNING 10-08 10:55:31 utils.py:564] Pin memory is not supported on XPU.
    INFO 10-08 10:55:31 gpu_executor.py:108] # GPU blocks: 1095, # CPU blocks: 512
    Processed prompts:   0%|                                                          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]Processed prompts: 100%|██████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.73s/it, est. speed input: 1.18 toks/s, output: 2.79 toks/s]Prompt: 'Hello, my name is', Generated text: ' [Your Name] and I am a [Your Profession] with [Your'
    Prompt: 'The president of the United States is', Generated text: ' the head of the executive branch of the federal government and is one of the most'
    Prompt: 'The capital of France is', Generated text: ' Paris, which is located in the northern part of the country.\nwhere is'
    Prompt: 'The future of AI is', Generated text: ' vast and exciting, with many possibilities for growth and innovation. Here are'
    ```

2. Online serving usage with `/llm/start-vllm-service.sh`
    1. Change the `/llm/start-vllm-service.sh`, set `model` parameter to gptq model path and `served_model_name`. Add `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

    ```bash
    #!/bin/bash
    model="/llm/models/Llama-2-7B-Chat-GPTQ/"
    served_model_name="llama2-7b-gptq"
    ...
    python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
      --served-model-name $served_model_name \
      --model $model \
      ...
      --quantization gptq \
      --load-in-low-bit asym_int4 \
      ...
    ```

    2. Use `bash start-vllm-service.sh` to start gptq model online serving. Serving start successfully log:

    ```bash
    2024-10-18 09:26:30,604 - INFO - Converting the current model to asym_int4 format......
    2024-10-18 09:26:30,605 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-18 09:26:35,970 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
    2024-10-18 09:26:37,007 - INFO - Loading model weights took 3.7381 GB
    WARNING 10-18 09:26:44 utils.py:564] Pin memory is not supported on XPU.
    INFO 10-18 09:26:44 gpu_executor.py:108] # GPU blocks: 2254, # CPU blocks: 1024
    WARNING 10-18 09:26:44 serving_embedding.py:171] embedding_mode is False. Embedding API will not work.
    INFO 10-18 09:26:44 launcher.py:14] Available routes are:
    INFO 10-18 09:26:44 launcher.py:22] Route: /openapi.json, Methods: GET, HEAD
    INFO 10-18 09:26:44 launcher.py:22] Route: /docs, Methods: GET, HEAD
    INFO 10-18 09:26:44 launcher.py:22] Route: /docs/oauth2-redirect, Methods: GET, HEAD
    INFO 10-18 09:26:44 launcher.py:22] Route: /redoc, Methods: GET, HEAD
    INFO 10-18 09:26:44 launcher.py:22] Route: /health, Methods: GET
    INFO 10-18 09:26:44 launcher.py:22] Route: /tokenize, Methods: POST
    INFO 10-18 09:26:44 launcher.py:22] Route: /detokenize, Methods: POST
    INFO 10-18 09:26:44 launcher.py:22] Route: /v1/models, Methods: GET
    INFO 10-18 09:26:44 launcher.py:22] Route: /version, Methods: GET
    INFO 10-18 09:26:44 launcher.py:22] Route: /v1/chat/completions, Methods: POST
    INFO 10-18 09:26:44 launcher.py:22] Route: /v1/completions, Methods: POST
    INFO 10-18 09:26:44 launcher.py:22] Route: /v1/embeddings, Methods: POST
    INFO:     Started server process [1294]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

    3. In docker send request to verfiy the serving status.

    ```bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "llama2-7b-gptq",
              "prompt": "San Francisco is a",
              "max_tokens": 128
            }'
    ```

    and should get following output:

    ```json
    {
      "id": "cmpl-e20bdfe80656404baea930e0288396a9",
      "object": "text_completion",
      "created": 1729214854,
      "model": "llama2-7b-gptq",
      "choices": [
        {
          "index": 0,
          "text": " food lover's paradise with a diverse array of culinary options to suit any taste and budget. Here are some of the top attractions when it comes to food and drink in San Francisco:\n\n1. Fisherman's Wharf: This bustling waterfront district is known for its fresh seafood, street performers, and souvenir shops. Be sure to try some of the local specialties like Dungeness crab, abalone, or sourdough bread.\n\n2. Chinatown: San Francisco's Chinatown is one of the largest and oldest",
          "logprobs": null,
          "finish_reason": "length",
          "stop_reason": null
        }
      ],
      "usage": {
        "prompt_tokens": 5,
        "total_tokens": 133,
        "completion_tokens": 128
      }
    }
    ```

### Advanced Features

#### Multi-modal Model

vLLM serving with IPEX-LLM supports multi-modal models, such as [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6), which can accept image and text input at the same time and respond.

1. Start MiniCPM service: change the `model` and `served_model_name` value in `/llm/start-vllm-service.sh`

2. Send request with image url and prompt text. (For successfully download image from url, you may need set `http_proxy` and `https_proxy` in docker before the vllm service started)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-2_6",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
```

3. Expect result should be like:

```bash
{"id":"chat-0c8ea64a2f8e42d9a8f352c160972455","object":"chat.completion","created":1728373105,"model":"MiniCPM-V-2_6","choices":[{"index":0,"message":{"role":"assistant","content":"这幅图片展示了一个小孩，可能是女孩，根据服装和发型来判断。她穿着一件有红色和白色条纹的连衣裙，一个可见的白色蝴蝶结，以及一个白色的 头饰，上面有红色的点缀。孩子右手拿着一个白色泰迪熊，泰迪熊穿着一个粉色的裙子，带有褶边，它的左脸颊上有一个红色的心形图案。背景模糊，但显示出一个自然户外的环境，可能是一个花园或庭院，有红花和石头墙。阳光照亮了整个场景，暗示这可能是正午或下午。整体氛围是欢乐和天真。","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":225,"total_tokens":353,"completion_tokens":128}}
```

#### Preifx Caching

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

1. Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example python script to show the time reduce of APC:

```python
import time
from vllm import SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM


# A prompt containing a large markdown table. The table is randomly generated by GPT-4.
LONG_PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n" + """
| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
|-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
| 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |
| 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |
| 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |
| 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |
| 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |
| 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |
| 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|
| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |
| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |
| 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |
| 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |
| 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |
| 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |
| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |
| 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |
| 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |
| 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |
| 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |
| 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |
| 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
| 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |
| 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |
| 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |
| 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |
| 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |
| 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |
| 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |
"""


def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")


# set enable_prefix_caching=True to enable APC
llm = LLM(model='/llm/models/Llama-2-7b-chat-hf',
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="fp8",
          tensor_parallel_size=1,
          max_model_len=2000,
          max_num_batched_tokens=2000,
          enable_prefix_caching=True)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Querying the age of John Doe
get_generation_time(
        llm,
        sampling_params,
        LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is ",
        )

# Querying the age of Zack Blue
# This query will be faster since vllm avoids computing the KV cache of LONG_PROMPT again.
get_generation_time(
        llm,
        sampling_params,
        LONG_PROMPT + "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is ",
        )

```

2. Expected output is shown as below: APC greatly reduces the generation time of the question related to the same table.

```bash
INFO 10-09 15:43:21 block_manager_v1.py:247] Automatic prefix caching is enabled.
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.97s/it, est. speed input: 84.57 toks/s, output: 0.73 toks/s]
Output: 29.
Question: What is the occupation of Jane Smith? Your answer
Generation time: 21.972806453704834 seconds.
Processed prompts: 100%|██████████████████████████████████████████████| 1/1 [00:00<00:00,  1.04it/s, est. speed input: 1929.67 toks/s, output: 16.63 toks/s]
Output: 30.
Generation time: 0.9657604694366455 seconds.
```

#### LoRA Adapter

This chapter shows how to use LoRA adapters with vLLM on top of a base model. Adapters can be efficiently served on a per request basis with minimal overhead.

1. Download the adapter(s) and save them locally first, for example, for `llama-2-7b`:

```bash
git clone https://huggingface.co/yard1/llama-2-7b-sql-lora-test
```

2. Start vllm server with LoRA adapter, setting `--enable-lora` and `--lora-modules` is necessary

```bash
export SQL_LOARA=your_sql_lora_model_path
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name Llama-2-7b-hf \
  --port 8000 \
  --model meta-llama/Llama-2-7b-hf \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --enable-lora \
  --lora-modules sql-lora=$SQL_LOARA
```

3. Send a request to sql-lora

```bash
curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
     "model": "sql-lora",
     "prompt": "San Francisco is a",
     "max_tokens": 128,
     "temperature": 0
     }'
```

4. Result expected show below:

```json
{
    "id": "cmpl-d6fa55b2bc404628bd9c9cf817326b7e",
    "object": "text_completion",
    "created": 1727367966,
    "model": "Llama-2-7b-hf",
    "choices": [
        {
            "index": 0,
            "text": " city in Northern California that is known for its vibrant cultural scene, beautiful architecture, and iconic landmarks like the Golden Gate Bridge and Alcatraz Island. Here are some of the best things to do in San Francisco:\n\n1. Explore Golden Gate Park: This sprawling urban park is home to several museums, gardens, and the famous Japanese Tea Garden. It's a great place to escape the hustle and bustle of the city and enjoy some fresh air and greenery.\n2. Visit Alcatraz Island: Take a ferry to the former prison and",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 133,
        "completion_tokens": 128
    }
}
```

5. For multi lora adapters, modify the sever start script's `--lora-modules` like this:

```bash
export SQL_LOARA_1=your_sql_lora_model_path_1
export SQL_LOARA_2=your_sql_lora_model_path_2
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  #other codes...
  --enable-lora \
  --lora-modules sql-lora-1=$SQL_LOARA_1 sql-lora-2=$SQL_LOARA_2

```

#### OpenAI API Backend

vLLM Serving can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as backend for web applications such as [open-webui](https://github.com/open-webui/open-webui/) using OpenAI API.

1. Start vLLM Serving with `api-key`, just setting any string to `api-key` in `start-vllm-service.sh`, and run it.

```bash
#!/bin/bash
model="/llm/models/Qwen1.5-14B-Chat"
served_model_name="Qwen1.5-14B-Chat"

#export SYCL_CACHE_PERSISTENT=1
export CCL_WORKER_COUNT=4
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4000 \
  --api-key <your-api-key> \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray
```

2. Send http request with `api-key` header to verify the model has deployed successfully.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <your-api-key>" \
    -d '{
    "model": "Qwen1.5-14B-Chat",
    "prompt": "San Francisco is a",
    "max_tokens": 128
    }'
```

3. Start open-webui serving with following scripts. Note that the `OPENAI_API_KEY` must be consistent with the backend value. The `<host-ip>` in `OPENAI_API_BASE_URL` is the ipv4 address of the host that starts docker. For relevant details, please refer to official document [link](https://docs.openwebui.com/#installation-for-openai-api-usage-only) of open-webui.

```bash
#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=<your-docker-container-name>

docker rm -f $CONTAINER_NAME

docker run -itd \
           -p 3000:8080 \
           -e OPENAI_API_KEY=<your-api-key> \
           -e OPENAI_API_BASE_URL=http://<host-ip>:8000/v1 \
           -v open-webui:/app/backend/data \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE  
```

Then you should start the docker on host that make sure you can visit vLLM backend serving.

4. After installation, you can access Open WebUI at <http://localhost:3000>. Enjoy! 😄

#### Serving with FastChat

We can set up model serving using `IPEX-LLM` as backend using FastChat, the following steps gives an example of how to deploy a demo using FastChat.

1. **Start the Docker Container**

    Run the following command to launch a Docker container with device access:

    ```bash
    #/bin/bash
    export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest

    sudo docker run -itd \
            --net=host \
            --device=/dev/dri \
            --name=demo-container \
            # Example: map host model directory to container
            -v /LLM_MODELS/:/llm/models/ \  
            --shm-size="16g" \
            # Optional: set proxy if needed
            -e http_proxy=... \ 
            -e https_proxy=... \
            -e no_proxy="127.0.0.1,localhost" \
            $DOCKER_IMAGE
    ```

2. **Start the FastChat Service**

    Enter the container and start the FastChat service:

    ```bash
    #/bin/bash

    # This command assumes that you have mapped the host model directory to the container
    # and the model directory is /llm/models/
    # we take Yi-1.5-34B as an example, and you can replace it with your own model

    ps -ef | grep "fastchat" | awk '{print $2}' | xargs kill -9
    pip install -U gradio==4.43.0
    
    # start controller
    python -m fastchat.serve.controller &

    export USE_XETLA=OFF
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
    
    export TORCH_LLM_ALLREDUCE=0
    export CCL_DG2_ALLREDUCE=1
    # CCL needed environment variables
    export CCL_WORKER_COUNT=4
    # pin ccl worker to cores
    # export CCL_WORKER_AFFINITY=32,33,34,35
    export FI_PROVIDER=shm
    export CCL_ATL_TRANSPORT=ofi
    export CCL_ZE_IPC_EXCHANGE=sockets
    export CCL_ATL_SHM=1
    
    source /opt/intel/1ccl-wks/setvars.sh
    
    python -m ipex_llm.serving.fastchat.vllm_worker \
    --model-path /llm/models/Yi-1.5-34B \
    --device xpu \
    --enforce-eager \
    --disable-async-output-proc \
    --distributed-executor-backend ray \
    --dtype float16 \
    --load-in-low-bit fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-batched-tokens 8000 &
    
    sleep 120
    
    python -m fastchat.serve.gradio_web_server &
    ```

This quick setup allows you to deploy FastChat with IPEX-LLM efficiently.

### Validated Models List

| models (fp8)     | gpus  |
| ---------------- | :---: |
| llama-3-8b       |   1   |
| Llama-2-7B       |   1   |
| Qwen2-7B         |   1   |
| Qwen1.5-7B       |   1   |
| GLM4-9B          |   1   |
| chatglm3-6b      |   1   |
| Baichuan2-7B     |   1   |
| Codegeex4-all-9b |   1   |
| Llama-2-13B      |   2   |
| Qwen1.5-14b      |   2   |
| TeleChat-13B     |   2   |
| Qwen1.5-32b      |   4   |
| Yi-1.5-34B       |   4   |
| CodeLlama-34B    |   4   |
