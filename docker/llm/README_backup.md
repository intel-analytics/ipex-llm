# Getting started with IPEX-LLM in Docker

This guide provides step-by-step instructions for installing and using IPEX-LLM in a Docker environment. It covers setups for both CPU and XPU (accelerated processing units) on different operating systems.

### Index
- [Docker Installation](#docker-installation-instructions)
- [IPEX-LLM Inference](#ipex-llm-inference-on-cpu)
    - [On CPU](#ipex-llm-inference-on-cpu)
    - [On XPU](#ipex-llm-inference-on-xpu)
- [IPEX-LLM Serving](#ipex-llm-serving-on-cpu)
    - [On CPU](#ipex-llm-serving-on-cpu)
    - [On XPU](#ipex-llm-serving-on-xpu)
- [IPEX-LLM Fine Tuning](#ipex-llm-fine-tuning-on-cpu)
    - [On CPU](#ipex-llm-fine-tuning-on-cpu)
    - [On XPU](#ipex-llm-fine-tuning-on-xpu)


## Docker Installation Instructions

1. **For New Users:**
   - Begin by visiting the [official Docker Get Started page](https://www.docker.com/get-started/) for a comprehensive introduction and installation guide.

2. **Additional Steps for Windows Users:**
   - For Windows Users, follow the step-by-step guide: [Docker Installation Instructions for Windows](https://github.com/intel-analytics/ipex-llm/blob/main/docs/readthedocs/source/doc/LLM/Quickstart/docker_windows_gpu.md).


## IPEX-LLM Inference on CPU

### 1. Prepare ipex-llm-cpu Docker Image

Run the following command to pull image:
```bash
docker pull intelanalytics/ipex-llm-cpu:2.2.0-SNAPSHOT
```

### 2. Start bigdl-llm-cpu Docker Container

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpu:2.2.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

docker run -itd \
    -p 12345:12345 \
    --cpuset-cpus="0-47" \
    --cpuset-mems="0" \
    --name=$CONTAINER_NAME \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE
```

Access the container:
```
docker exec -it $CONTAINER_NAME bash
```

### 3. Start Inference and Tutorials
**3.1 Chat Interface**: Use `chat.py` for conversational AI. 

For example, if your model is Llama-2-7b-chat-hf and mounted on /llm/models, you can excute the following command to initiate a conversation:
  ```bash
  cd /llm/portable-zip
  python chat.py --model-path /llm/models/Llama-2-7b-chat-hf
  ```
Here is a demostration:

<a align="left"  href="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-chatpy-demo.gif">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-chatpy-demo.gif" width='60%' /> 

</a><br>

**3.2 Jupyter Lab Tutorials**: Start a Jupyter Lab session for IPEX-LLM tutorials.

Run the following command to start notebook:
```bash
cd /llm
./start-notebook.sh [--port EXPECTED_PORT]
```
The default port is 12345, you could assign a different port by specifying the --port parameter.

If you're using the host network mode when booting the container, once the service is running successfully, you can access the tutorial at http://127.0.0.1:12345/lab. Alternatively, you need to ensure the correct ports are bound between the container and the host. 

Here's a demonstration of how to navigate the tutorial in the explorer:

<a align="left" href="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-tutorial-demo.gif">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-tutorial-demo.gif" width='60%' /> 

</a> <br>

**3.3 Performance Benchmark**: We provide a benchmark tool help users to test all the benchmarks and record them in a result CSV. 

```bash
cd /llm/benchmark/all-in-one
```

Users can provide models and related information in config.yaml.
```bash
repo_id:
  # - 'THUDM/chatglm-6b'
  # - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
  # - 'liuhaotian/llava-v1.5-7b' # requires a LLAVA_REPO_DIR env variables pointing to the llava dir; added only for gpu win related test_api now
local_model_hub: 'path to your local model hub'
warm_up: 1
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  # - "transformer_int4"
  # - "native_int4"
  # - "optimize_model"
  # - "pytorch_autocast_bf16"
  # - "transformer_autocast_bf16"
  # - "bigdl_ipex_bf16"
  # - "bigdl_ipex_int4"
  # - "bigdl_ipex_int8"
  # - "ipex_fp16_gpu" # on Intel GPU
  # - "bigdl_fp16_gpu" # on Intel GPU
  # - "transformer_int4_gpu"  # on Intel GPU
  # - "optimize_model_gpu"  # on Intel GPU
  # - "deepspeed_transformer_int4_cpu" # on Intel SPR Server
  # - "transformer_int4_gpu_win" # on Intel GPU for Windows
  # - "transformer_int4_fp16_gpu_win" # on Intel GPU for Windows, use fp16 for non-linear layer
  # - "transformer_int4_loadlowbit_gpu_win" # on Intel GPU for Windows using load_low_bit API. Please make sure you have used the save.py to save the converted low bit model
  # - "deepspeed_optimize_model_gpu" # deepspeed autotp on Intel GPU
  - "speculative_cpu"
  # - "speculative_gpu"
cpu_embedding: False # whether put embedding to CPU (only avaiable now for gpu win related test_api)
streaming: False # whether output in streaming way (only avaiable now for gpu win related test_api)
```

This benchmark tool offers various test APIs, including `transformer_int4`, `speculative_cpu`, and more.

For instance, if you wish to benchmark **inference with speculative decoding**, utilize the `speculative_cpu` test API in the `config.yml` file. 

Then, execute `bash run-spr.sh`, which will generate output results in `results.csv`.
```bash
bash run-spr.sh
```

For further details and comprehensive functionality of the benchmark tool, please refer to the [all-in-one benchmark tool](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/dev/benchmark/all-in-one).

Additionally, for examples related to Inference with Speculative Decoding, you can explore [Speculative-Decoding Examples](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding).



## IPEX-LLM Inference on XPU

### 1. Prepare ipex-llm-xpu Docker Image

Run the following command to pull image from dockerhub:
```bash
docker pull intelanalytics/ipex-llm-xpu:2.2.0-SNAPSHOT
```

### 2. Start Chat Inference

We provide `chat.py` for conversational AI. If your model is Llama-2-7b-chat-hf and mounted on /llm/models, you can execute the following command to initiate a conversation:

To map the xpu into the container, you need to specify --device=/dev/dri when booting the container.

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.2.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models \
        $DOCKER_IMAGE bash -c "python chat.py --model-path /llm/models/Llama-2-7b-chat-hf"
```


### 3. Quick Performance Benchmark

Execute a quick performance benchmark by starting the ipex-llm-xpu container, specifying the model, test API, and device, then running the benchmark.sh script. 

To map the XPU into the container, specify `--device=/dev/dri` when booting the container.
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.2.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models [change to your model path]

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models \
        -e REPO_IDS="meta-llama/Llama-2-7b-chat-hf" \
        -e TEST_APIS="transformer_int4_gpu" \
        -e DEVICE=Arc \
        $DOCKER_IMAGE /llm/benchmark.sh
```

Customize environment variables to specify:

- **REPO_IDS:** Model's name and organization, separated by commas if multiple values exist.
- **TEST_APIS:** Different test functions based on the machine, separated by commas if multiple values exist.
- **DEVICE:** Type of device - Max, Flex, Arc.

**Result**

Upon completion, you can obtain a CSV result file, the content of CSV results will be printed out. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results.

## IPEX-LLM Serving on CPU
FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

IPEX-LLM is integrated into FastChat so that user can use IPEX-LLM as a serving backend in the deployment.

### 1. Prepare ipex-llm-serving-cpu Docker Image

Run the following command:

```bash
docker pull intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT
```

### 2. Start ipex-llm-serving-cpu Docker Container

Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.

```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

docker run -itd \
    --net=host \
    --cpuset-cpus="0-47" \
    --cpuset-mems="0" \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE
```
Access the container:
```
docker exec -it $CONTAINER_NAME bash
```

### 3. Serving with FastChat

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

- #### **Step 1: Launch the Controller**
  ```bash
  python3 -m fastchat.serve.controller &
  ```

  This controller manages the distributed workers.

- #### **Step 2: Launch the model worker(s)**

  Using IPEX-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.
  ```bash
  source ipex-llm-init -t

  # Available low_bit format including sym_int4, sym_int8, bf16 etc.
  python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path path/to/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "cpu" &
  ```
  Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

- #### **Step 3: Launch Gradio web server or RESTful API server**
  You can launch Gradio web server to serve your models using the web UI or launch RESTful API server to serve with HTTP.

  - **Option 1: Serving with Web UI**
    ```bash
    python3 -m fastchat.serve.gradio_web_server &
    ```
    This is the user interface that users will interact with.

    By following these steps, you will be able to serve your models using the web UI with `IPEX-LLM` as the backend. You can open your browser and chat with a model now.

  - **Option 2: Serving with OpenAI-Compatible RESTful APIs**

      Launch the RESTful API server

      ```bash
      python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000 &
      ```

      Use curl for testing, an example could be:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{
        "model": "Llama-2-7b-chat-hf",
        "prompt": "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
        "n": 1,
        "best_of": 1,
        "use_beam_search": false,
        "stream": false
      }' http://YOUR_HTTP_HOST:8000/v1/completions
      ```
  You can find more details here [Serving using IPEX-LLM and FastChat](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/serving/fastchat/README.md)

### 4. Serving with vLLM Continuous Batching
To fully utilize the continuous batching feature of the vLLM, you can send requests to the service using curl or other similar methods. The requests sent to the engine will be batched at token level. Queries will be executed in the same forward step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.

- #### **Step 1: Launch the api_server**
  ```bash
  #!/bin/bash
  # You may also want to adjust the `--max-num-batched-tokens` argument, it indicates the hard limit
  # of batched prompt length the server will accept
  numactl -C 0-47 -m 0 python -m ipex_llm.vllm.entrypoints.openai.api_server \
          --model /llm/models/Llama-2-7b-chat-hf/  \
          --host 0.0.0.0 --port 8000 \
          --load-format 'auto' --device cpu --dtype bfloat16 \
          --max-num-batched-tokens 4096 &
  ```

- #### **Step 2: Use curl for testing, access the api server as follows:**

  ```bash
  curl http://YOUR_HTTP_HOST:8000/v1/completions \
          -H "Content-Type: application/json" \
          -d '{
                  "model": "/llm/models/Llama-2-7b-chat-hf/",
                  "prompt": "San Francisco is a",
                  "max_tokens": 128,
                  "temperature": 0
  }' &
  ```

  You can find more details here: [Serving with vLLM Continuous Batching](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/vLLM-Serving/README.md)


## IPEX-LLM Serving on XPU

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

IPEX-LLM is integrated into FastChat so that user can use IPEX-LLM as a serving backend in the deployment.

### 1. Prepare ipex-llm-serving-xpu Docker Image

Run the following command:

```bash
docker pull intelanalytics/ipex-llm-serving-xpu:2.2.0-SNAPSHOT
```

### 2. Start ipex-llm-serving-xpu Docker Container

To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container.

```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.2.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

docker run -itd \
    --net=host \
    --cpuset-cpus="0-47" \
    --cpuset-mems="0" \
    --name=$CONTAINER_NAME \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE
```
Access the container:
```
docker exec -it $CONTAINER_NAME bash
```
To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

### 3. Serving with FastChat

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

- #### **Step 1: Launch the Controller**
  ```bash
  python3 -m fastchat.serve.controller &
  ```

  This controller manages the distributed workers.

- #### **Step 2: Launch the model worker(s)**

  Using IPEX-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.
  ```bash
  # Available low_bit format including sym_int4, sym_int8, fp16 etc.
  python3 -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path /llm/models/Llama-2-7b-chat-hf/ --low-bit "sym_int4" --trust-remote-code --device "xpu" &
  ```
  Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

- #### **Step 3: Launch Gradio web server or RESTful API server**
  You can launch Gradio web server to serve your models using the web UI or launch RESTful API server to serve with HTTP.

  - **Option 1: Serving with Web UI**
    ```bash
    python3 -m fastchat.serve.gradio_web_server &
    ```
    This is the user interface that users will interact with.

    By following these steps, you will be able to serve your models using the web UI with `IPEX-LLM` as the backend. You can open your browser and chat with a model now.

  - **Option 2: Serving with OpenAI-Compatible RESTful APIs**

      Launch the RESTful API server

      ```bash
      python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000 &
      ```

      Use curl for testing, an example could be:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{
        "model": "Llama-2-7b-chat-hf",
        "prompt": "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
        "n": 1,
        "best_of": 1,
        "use_beam_search": false,
        "stream": false
      }' http://YOUR_HTTP_HOST:8000/v1/completions
      ```
  You can find more details here [Serving using IPEX-LLM and FastChat](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/serving/fastchat/README.md)

### 4. Serving with vLLM Continuous Batching
To fully utilize the continuous batching feature of the vLLM, you can send requests to the service using curl or other similar methods. The requests sent to the engine will be batched at token level. Queries will be executed in the same forward step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.

- #### **Step 1: Launch the api_server**
  ```bash
  #!/bin/bash
  # You may also want to adjust the `--max-num-batched-tokens` argument, it indicates the hard limit
  # of batched prompt length the server will accept
  python -m ipex_llm.vllm.entrypoints.openai.api_server \
          --model /llm/models/Llama-2-7b-chat-hf/ \
          --host 0.0.0.0 --port 8000 \
          --load-format 'auto' --device xpu --dtype bfloat16 \
          --max-num-batched-tokens 4096 &
  ```

- #### **Step 2: Use curl for testing, access the api server as follows:**

  ```bash
  curl http://YOUR_HTTP_HOST:8000/v1/completions \
          -H "Content-Type: application/json" \
          -d '{
                  "model": "/llm/models/Llama-2-7b-chat-hf/",
                  "prompt": "San Francisco is a",
                  "max_tokens": 128,
                  "temperature": 0
  }' &
  ```
  You can find more details here [Serving with vLLM Continuous Batching](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md)

## IPEX-LLM Fine Tuning on CPU

### 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
# For standalone
docker pull intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.2.0-SNAPSHOT

# For k8s
docker pull intelanalytics/ipex-llm-finetune-qlora-cpu-k8s:2.2.0-SNAPSHOT
```

Or build the image from source:

```bash
# For standalone
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.2.0-SNAPSHOT \
  -f ./Dockerfile .

# For k8s
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-qlora-cpu-k8s:2.2.0-SNAPSHOT \
  -f ./Dockerfile.k8s .
```

### 2. Prepare Base Model, Data and Container

Here, we try to fine-tune a [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, and please download them and start a docker container with files mounted like below:

```bash
export BASE_MODE_PATH=your_downloaded_base_model_path
export DATA_PATH=your_downloaded_data_path
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --name=ipex-llm-fintune-qlora-cpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/ipex_llm/model \
   -v $DATA_PATH:/ipex_llm/data/alpaca-cleaned \
   intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.2.0-SNAPSHOT
```

The download and mount of base model and data to a docker container demonstrates a standard fine-tuning process. You can skip this step for a quick start, and in this way, the fine-tuning codes will automatically download the needed files:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --name=ipex-llm-fintune-qlora-cpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.2.0-SNAPSHOT
```

However, we do recommend you to handle them manually, because the automatical download can be blocked by Internet access and Huggingface authentication etc. according to different environment, and the manual method allows you to fine-tune in a custom way (with different base model and dataset).

### 3. Start Fine-Tuning (Local Mode)

Enter the running container:

```bash
docker exec -it ipex-llm-fintune-qlora-cpu bash
```

Then, start QLoRA fine-tuning:
If the machine memory is not enough, you can try to set `use_gradient_checkpointing=True`.

```bash
cd /ipex_llm
bash start-qlora-finetuning-on-cpu.sh
```

After minutes, it is expected to get results like:

```bash
{'loss': 2.0251, 'learning_rate': 0.0002, 'epoch': 0.02}
{'loss': 1.2389, 'learning_rate': 0.00017777777777777779, 'epoch': 0.03}
{'loss': 1.032, 'learning_rate': 0.00015555555555555556, 'epoch': 0.05}
{'loss': 0.9141, 'learning_rate': 0.00013333333333333334, 'epoch': 0.06}
{'loss': 0.8505, 'learning_rate': 0.00011111111111111112, 'epoch': 0.08}
{'loss': 0.8713, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.09}
{'loss': 0.8635, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.11}
{'loss': 0.8853, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.12}
{'loss': 0.859, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.14}
{'loss': 0.8608, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15}
100%|███████████████████████████████████████████████████████████████████████████████████| 200/200 [07:16<00:00,  2.18s/it]
TrainOutput(global_step=200, training_loss=1.0400420665740966, metrics={'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15})
```

### 4. Merge the adapter into the original model

Using the [export_merged_model.py](../../../../../../python/llm/example/GPU/LLM-Finetuning/QLoRA/export_merged_model.py) to merge.

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.


## IPEX-LLM Fine Tuning on XPU

The following shows how to fine-tune LLM with Quantization (QLoRA built on IPEX-LLM 4bit optimizations) in a docker environment, which is accelerated by Intel XPU.

### 1. Prepare ipex-llm-finetune-xpu Docker Image

Run the following command:

```bash
docker pull intelanalytics/ipex-llm-finetune-xpu:2.2.0-SNAPSHOT
```

### 2. Prepare Base Model, Data and Start Docker Container

Here, we try to fine-tune a [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, and please download them and start a docker container with files mounted like below:

```bash
export BASE_MODE_PATH=your_downloaded_base_model_path
export DATA_PATH=your_downloaded_data_path
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export CONTAINER_NAME=my_container
export DOCKER_IMAGE=intelanalytics/ipex-llm-finetune-xpu:2.2.0-SNAPSHOT

docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=$CONTAINER_NAME \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   $DOCKER_IMAGE
```

After the container is booted, you could get into the container through docker exec.

```bash
docker exec -it $CONTAINER_NAME bash
```

### 3. Start Fine-Tuning (Local Mode)

Then, start QLoRA fine-tuning:

```bash
bash start-qlora-finetuning-on-xpu.sh
```

After minutes, it is expected to get results like:

```bash
{'loss': 2.0251, 'learning_rate': 0.0002, 'epoch': 0.02}
{'loss': 1.2389, 'learning_rate': 0.00017777777777777779, 'epoch': 0.03}
{'loss': 1.032, 'learning_rate': 0.00015555555555555556, 'epoch': 0.05}
{'loss': 0.9141, 'learning_rate': 0.00013333333333333334, 'epoch': 0.06}
{'loss': 0.8505, 'learning_rate': 0.00011111111111111112, 'epoch': 0.08}
{'loss': 0.8713, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.09}
{'loss': 0.8635, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.11}
{'loss': 0.8853, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.12}
{'loss': 0.859, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.14}
{'loss': 0.8608, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15}
100%|███████████████████████████████████████████████████████████████████████████████████| 200/200 [07:16<00:00,  2.18s/it]
TrainOutput(global_step=200, training_loss=1.0400420665740966, metrics={'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15})
```

### 4. Merge the adapter into the original model

Using the [export_merged_model.py](../../python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora/export_merged_model.py) to merge.

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
