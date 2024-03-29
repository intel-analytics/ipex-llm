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

Run the following command to pull image from dockerhub:
```bash
docker pull intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT
```

### 2. Start bigdl-llm-cpu Docker Container

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

docker run -itd \
    --privileged \
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

For example, if your model is chatglm-6b and mounted on /llm/models, you can excute the following command to initiate a conversation:
  ```bash
  cd /llm/portable-zip
  python chat.py --model-path /llm/models/chatglm2-6b
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
cd /llm//benchmark/all-in-one
```

Users can provide models and related information in config.yaml.
```bash
repo_id:
  - 'THUDM/chatglm-6b'
  - 'THUDM/chatglm2-6b'
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
  - "transformer_int4"
  - "native_int4"
  - "optimize_model"
  - "pytorch_autocast_bf16"
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
  # - "speculative_cpu"
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

For further details and comprehensive functionality of the benchmark tool, please refer to the [all-in-one benchmark tool](https://github.com/intel-analytics/BigDL/tree/main/python/llm/dev/benchmark/all-in-one).

Additionally, for examples related to Inference with Speculative Decoding, you can explore [Speculative-Decoding Examples](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/Speculative-Decoding).



## IPEX-LLM Inference on XPU

First, pull docker image from docker hub:
```
docker pull intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```
To map the xpu into the container, you need to specify --device=/dev/dri when booting the container.
An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models \
        $DOCKER_IMAGE
```

After the container is booted, you could get into the container through `docker exec`.

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

To run inference using `IPEX-LLM` using xpu, you could refer to this [documentation](https://github.com/intel-analytics/IPEX/tree/main/python/llm/example/GPU).

## IPEX-LLM Serving on CPU

### Boot container

Pull image:
```
docker pull intelanalytics/ipex-llm-serving-cpu:2.1.0-SNAPSHOT
```

You could use the following bash script to start the container. Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.
```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.1.0-SNAPSHOT
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
After the container is booted, you could get into the container through `docker exec`.

### Models

Using IPEX-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

FastChat determines the Model adapter to use through path matching. Therefore, in order to load models using IPEX-LLM, you need to make some modifications to the model's name.

A special case is `ChatGLM` models. For these models, you do not need to do any changes after downloading the model and the `IPEX-LLM` backend will be used automatically.


### Start the service

#### Serving with Web UI

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

##### Launch the Controller
```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

##### Launch the model worker(s)
```bash
python3 -m ipex_llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

> To run model worker using Intel GPU, simply change the --device cpu option to --device xpu

##### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with `IPEX-LLM` as the backend. You can open your browser and chat with a model now.

#### Serving with OpenAI-Compatible RESTful APIs

To start an OpenAI API server that provides compatible APIs using `IPEX-LLM` backend, you need three main components: an OpenAI API Server that serves the in-coming requests, model workers that host one or more models, and a controller to coordinate the web server and model workers.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s):

```bash
python3 -m ipex_llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```


## IPEX-LLM Serving on XPU

### Boot container

Pull image:
```
docker pull intelanalytics/ipex-llm-serving-xpu:2.1.0-SNAPSHOT
```

To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container.

An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]
export SERVICE_MODEL_PATH=/llm/models/chatglm2-6b[a specified model path for running service]

docker run -itd \
    --net=host \
    --device=/dev/dri \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    --shm-size="16g" \
    -v $MODEL_PATH:/llm/models \
    -e SERVICE_MODEL_PATH=$SERVICE_MODEL_PATH \
    $DOCKER_IMAGE --service-model-path $SERVICE_MODEL_PATH
```
You can assign specified model path to service-model-path to run the service while booting the container. Also you can manually run the service after entering container. Run `/opt/entrypoint.sh --help` in container to see more information. There are steps below describe how to run service in details as well.

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```
After the container is booted, you could get into the container through `docker exec`.

### Start the service

#### Serving with Web UI

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

##### Launch the Controller
```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

##### Launch the model worker(s)
```bash
python3 -m ipex_llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

##### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with `IPEX-LLM` as the backend. You can open your browser and chat with a model now.

#### Serving with OpenAI-Compatible RESTful APIs

To start an OpenAI API server that provides compatible APIs using `IPEX-LLM` backend, you need three main components: an OpenAI API Server that serves the in-coming requests, model workers that host one or more models, and a controller to coordinate the web server and model workers.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s):

```bash
python3 -m ipex_llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

## IPEX-LLM Fine Tuning on CPU

### 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
# For standalone
docker pull intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT

# For k8s
docker pull intelanalytics/ipex-llm-finetune-qlora-cpu-k8s:2.1.0-SNAPSHOT
```

Or build the image from source:

```bash
# For standalone
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT \
  -f ./Dockerfile .

# For k8s
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-qlora-cpu-k8s:2.1.0-SNAPSHOT \
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
   intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT
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
   intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT
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

### 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT \
  -f ./Dockerfile .
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
   --device=/dev/dri \
   --memory="32G" \
   --name=ipex-llm-fintune-qlora-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   intelanalytics/ipex-llm-fintune-qlora-xpu:2.1.0-SNAPSHOT
```

The download and mount of base model and data to a docker container demonstrates a standard fine-tuning process. You can skip this step for a quick start, and in this way, the fine-tuning codes will automatically download the needed files:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=ipex-llm-fintune-qlora-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   --shm-size="16g" \
   intelanalytics/ipex-llm-fintune-qlora-xpu:2.1.0-SNAPSHOT
```

However, we do recommend you to handle them manually, because the automatical download can be blocked by Internet access and Huggingface authentication etc. according to different environment, and the manual method allows you to fine-tune in a custom way (with different base model and dataset).

### 3. Start Fine-Tuning

Enter the running container:

```bash
docker exec -it ipex-llm-fintune-qlora-xpu bash
```

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
