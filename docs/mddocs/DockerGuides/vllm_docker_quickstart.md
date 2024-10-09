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
    |`model="YOUR_MODEL"`| the model path in docker, for example "/llm/models/Llama-2-7b-chat-hf"|
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
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1
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

Use AWQ as a way to reduce memory footprint.

1. First download the model after awq quantification, taking `Llama-2-7B-Chat-AWQ` as an example, download it on <https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ>

2. Change the `/llm/vllm_offline_inference.py` LLM class code block's parameters `model`, `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

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

3. Expected result shows as below:

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

#### GPTQ

Use GPTQ as a way to reduce memory footprint.

1. First download the model after gptq quantification, taking `Llama-2-13B-Chat-GPTQ` as an example, download it on <https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ>

2. Change the `/llm/vllm_offline_inference` LLM class code block's parameters `model`, `quantization` and `load_in_low_bit`, note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`:

```python
llm = LLM(model="/llm/models/Llama-2-7B-Chat-GPTQ/",
          quantization="GPTQ",
          load_in_low_bit="asym_int4",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          tensor_parallel_size=1)
```

3. Expected result shows as below:

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

### Advanced Features

#### Multi-modal Model

vLLM serving with IPEX-LLM supports multi-modal models, such as [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6), which can accept image and text input at the same time and respond.

1. Start MiniCPM service: change the `model` and `served_model_name` value in `/llm/start-vllm-service.sh`

```bash
```

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

#### Preifx Caching[todo]

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
  --max-num-seqs 12 \
  --tensor-parallel-size 1 \
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

#### Cpu Offloading[todo]

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
