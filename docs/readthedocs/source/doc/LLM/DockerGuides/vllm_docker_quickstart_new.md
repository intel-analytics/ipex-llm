# vLLM Serving with IPEX-LLM on Intel GPUs via Docker

This guide demonstrates how to run `vLLM` serving with `IPEX-LLM` on Intel GPUs via Docker.

## Install docker

Follow the instructions in this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#linux) to install Docker on Linux.

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
    |`load_in_low_bit="fp8"`| model quantization accuracy, acceptable `fp8`, `fp6`, `sym_int4`, default is `fp8`|
    |`tensor_parallel_size=1`| number of graphics cards used by the model, default is `1`|

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

Before performing benchmark or starting the service, you can refer to this [section](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#runtime-configurations) to setup our recommended runtime configurations.

### Serving

#### Single card serving

A script named `/llm/start-vllm-service.sh` have been included in the image for starting the service conveniently. Here are the steps to use it.

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

For larger models (greater than 10b), we need to use multiple graphics cards for deployment. In the above single-card script(`/llm/start-vllm-service.sh`), we need to make some modifications to achieve multi-card serving.

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

3. **TP+PP Serving**: using tensor-parallel and pipline-parallel mixed, for example, using 2 cards for tp and 2 cards for pp serving, add following parameter:

```bash
--pipeline-parallel-size 2 \
--tensor-parallel-size 2
```

or shortening:

```bash
-pp 2 \
-tp 2
```

vLLM supports to utilize multiple cards through tensor parallel.

You can refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/vLLM_quickstart.html#about-tensor-parallel) on how to utilize the `tensor-parallel` feature and start the service.

### Quantization

The accuracy of the quantitative model is reduced from FP16 to INT4, which effectively reduces the file size by about 70 %. The main advantage is lower delay and memory usage.
Quantizing reduces the model’s precision from FP16 to INT4 which effectively reduces the file size by ~70%. The main benefits are lower latency and memory usage.

#### IPEX-LLM

Below shows an example output using `Qwen1.5-7B-Chat` with low-bit format `sym_int4`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" width=100%; />
</a>

#### AWQ

Use AWQ as a way to reduce memory footprint.

1. First download the model after awq quantification, taking `Llama-2-7B-Chat-AWQ` as an example, download it on <https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ>

2. Change the `/llm/vllm_offline_inference` LLM class code block's parameters `model`, `quantization` and `load_in_low_bit`:

```python
llm = LLM(model="/llm/models/Llama-2-7B-Chat-AWQ/",
          quantization="AWQ",
          load_in_low_bit="sym_int4",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          max_model_len=2000,
          max_num_batched_tokens=2000,
          tensor_parallel_size=1)
```

3. Expected result shows as below:
[TODO]: can't not run now???

#### GPTQ

### Advanced Features

#### Multi-modal Model

You can tune the service using these four arguments:

- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-batched-token`
- `--max-num-seq`

You can refer to this [doc](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/vLLM_quickstart.html#service) for a detailed explaination on these parameters.

#### Preifx Caching

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

#### Cpu Offloading

### Validated Models List
