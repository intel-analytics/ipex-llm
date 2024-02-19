# Getting started with BigDL-LLM in Docker

This guide provides step-by-step instructions for installing and using BigDL-LLM in a Docker environment. It covers setups for both CPU and XPU (accelerated processing units) on different operating systems.

### Index
- [Docker installation](#docker-installation-guide-for-bigdl-llm-on-cpu)

- [BigDL LLM Inference](#docker-installation-guide-for-bigdl-llm-on-xpu) 
    - [On CPU](#bigdl-llm-on-windows)
    - [On XPU](#bigdl-llm-on-linuxmacos)
- [BigDL LLM Serving](#docker-installation-guide-for-bigdl-llm-serving-on-cpu) 
    - [On CPU](#bigdl-llm-on-windows)
    - [On XPU](#bigdl-llm-on-linuxmacos)
- [BigDL LLM Fine Tuning](#docker-installation-guide-for-bigdl-llm-fine-tuning-on-cpu) 
    - [On CPU](#bigdl-llm-on-windows)
    - [On XPU](#bigdl-llm-on-linuxmacos)


## Docker Installation Instructions

**Getting Started with Docker:**

1. **For New Users:**
   - Begin by visiting the [official Docker Get Started page](https://www.docker.com/get-started/) for a comprehensive introduction and installation guide.

2. **Additional Steps for Windows Users:**
   - Ensure that WSL2 (Windows Subsystem for Linux version 2) or Hyper-V is enabled on your system. This is a prerequisite for running Docker on Windows.
   - Detailed installation instructions for Windows, including steps for enabling WSL2 or Hyper-V, can be found on the [Docker Desktop for Windows installation page](https://docs.docker.com/desktop/install/windows-install/).

By following these steps, you'll have Docker installed and ready for use on your machine.

## BigDL LLM Inference on CPU

**Step 1: Pull the bigdl-llm-cpu Image**
   - Run the following command:
     ```bash
     docker pull intelanalytics/bigdl-llm-cpu:2.5.0-SNAPSHOT
     ```


**Step 2: Start the bigdl-llm-cpu Container**
   - On Windows, create and run a batch script (`*.bat`) with the following content:
     ```bat
     @echo off
     set DOCKER_IMAGE=intelanalytics/bigdl-llm-cpu:2.5.0-SNAPSHOT
     set CONTAINER_NAME=my_container
     set MODEL_PATH=D:/llm/models[change to your model path]

     :: Run the Docker container
     docker run -itd ^
         -p 12345:12345 ^
         --cpuset-cpus="0-7" ^
         --cpuset-mems="0" ^
         --memory="8G" ^
         --name=%CONTAINER_NAME% ^
         -v %MODEL_PATH%:/llm/models ^
         %DOCKER_IMAGE%
     ```
   - On Linux/MacOS, the instructions for are similar to Windows, with the following script for starting the container:
      ```bash
      #/bin/bash
      export DOCKER_IMAGE=intelanalytics/bigdl-llm-cpu:2.5.0-SNAPSHOT
      export CONTAINER_NAME=my_container
      export MODEL_PATH=/llm/models[change to your model path]

      docker run -itd \
          -p 12345:12345 \
          --cpuset-cpus="0-7" \
          --cpuset-mems="0" \
          --memory="8G" \
          --name=$CONTAINER_NAME \
          -v $MODEL_PATH:/llm/models \
          $DOCKER_IMAGE
      ```
   - Access the container:
     ```
     docker exec -it my_container bash
     ```

**Step 3: Running Inference and Tutorials**
   - **Chat Interface**: Use `chat.py` for conversational AI. For example:
     ```bash
     cd /llm/portable-zip
     python chat.py --model-path /llm/models/chatglm-6b
     ```
   - **Jupyter Lab Tutorials**: Start a Jupyter Lab session for BigDL-LLM tutorials.
     ```bash
     cd /llm
     ./start-notebook.sh [--port EXPECTED_PORT]
     ```
     Access the tutorials at http://127.0.0.1:12345/lab.



## BigDL LLM Inference on XPU
**Step 1: Pull the bigdl-llm-xpu Image**
   - Run the following command:
     ```bash
     docker pull intelanalytics/bigdl-llm-xpu:2.5.0-SNAPSHOT
     ```


**Step 2: Start the bigdl-llm-xpu Container**
   - To map the xpu into the container, you need to specify --device=/dev/dri when booting the container. An example could be:
      ```bash
      #/bin/bash
      export DOCKER_IMAGE=intelanalytics/bigdl-llm-xpu:2.5.0-SNAPSHOT
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

**Step 3: Running Inference**
  - To run inference using `BigDL-LLM` using xpu, you could refer to this [documentation](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/GPU).

## BigDL LLM Serving on CPU

**Step 1: Pull the bigdl-llm-serving-cpu Image**
   - Run the following command:
     ```bash
     docker pull intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
     ```


**Step 2: Start the bigdl-llm-serving-cpu Container**
 - Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.

    ```bash
    export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
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
    ```
    docker exec
    ```


**Step 3: Prepare Models**

  - Using BigDL-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

  - FastChat determines the Model adapter to use through path matching. Therefore, in order to load models using BigDL-LLM, you need to make some modifications to the model's name.

  - For instance, assuming you have downloaded the `llama-7b-hf` from [HuggingFace](https://huggingface.co/decapoda-research/llama-7b-hf).  Then, to use the `BigDL-LLM` as backend, you need to change name from `llama-7b-hf` to `bigdl-7b`.
  The key point here is that the model's path should include "bigdl" and should not include paths matched by other model adapters.

  - A special case is `ChatGLM` models. For these models, you do not need to do any changes after downloading the model and the `BigDL-LLM` backend will be used automatically.


**Step 4: Start the service**

**Option 1: Serving with Web UI**

To serve using the Web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the web server and model workers.

- **Launch the Controller**
  ```bash
  python3 -m fastchat.serve.controller
  ```

  This controller manages the distributed workers.

- **Launch the model worker(s)**
  ```bash
  python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
  ```
  Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

  > To run model worker using Intel GPU, simply change the --device cpu option to --device xpu

- **Launch the Gradio web server**

  ```bash
  python3 -m fastchat.serve.gradio_web_server
  ```

  This is the user interface that users will interact with.

  By following these steps, you will be able to serve your models using the web UI with `BigDL-LLM` as the backend. You can open your browser and chat with a model now.

**Option 2: Serving with OpenAI-Compatible RESTful APIs**

To start an OpenAI API server that provides compatible APIs using `BigDL-LLM` backend, you need three main components: an OpenAI API Server that serves the in-coming requests, model workers that host one or more models, and a controller to coordinate the web server and model workers.

- **Launch the Controller**
  ```bash
  python3 -m fastchat.serve.controller
  ```

- **Launch the model worker(s)**

  ```bash
  python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device cpu
  ```

- **Launch the RESTful API server**

  ```bash
  python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
  ```


## BigDL LLM Serving on XPU

### Boot container

Pull image:
```
docker pull intelanalytics/bigdl-llm-serving-xpu:2.5.0-SNAPSHOT
```

To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container.

An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
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
python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

##### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with `BigDL-LLM` as the backend. You can open your browser and chat with a model now.

#### Serving with OpenAI-Compatible RESTful APIs

To start an OpenAI API server that provides compatible APIs using `BigDL-LLM` backend, you need three main components: an OpenAI API Server that serves the in-coming requests, model workers that host one or more models, and a controller to coordinate the web server and model workers.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s):

```bash
python3 -m bigdl.llm.serving.model_worker --model-path lmsys/vicuna-7b-v1.3 --device xpu
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

## BigDL LLM Fine Tuning on CPU

### 1. Prepare BigDL image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/bigdl-llm-finetune-lora-cpu:2.5.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-lora-cpu:2.5.0-SNAPSHOT \
  -f ./Dockerfile .
```

### 2. Prepare Base Model, Data and Container

Here, we try to finetune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [Cleaned alpaca data](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json), which contains all kinds of general knowledge and has already been cleaned. And please download them and start a docker container with files mounted like below:

```
docker run -itd \
 --name=bigdl-llm-fintune-lora-cpu \
 --cpuset-cpus="your_expected_range_of_cpu_numbers" \
 -e STANDALONE_DOCKER=TRUE \
 -e WORKER_COUNT_DOCKER=your_worker_count \
 -v your_downloaded_base_model_path:/bigdl/model \
 -v your_downloaded_data_path:/bigdl/data/alpaca_data_cleaned_archive.json \
 intelanalytics/bigdl-llm-finetune-cpu:2.5.0-SNAPSHOT \
 bash
```

You can adjust the configuration according to your own environment. After our testing, we recommend you set worker_count=1, and then allocate 80G memory to Docker.

### 3. Start Finetuning

Enter the running container:

```
docker exec -it bigdl-llm-fintune-lora-cpu bash
```

Then, run the script to start finetuning:

```
bash /bigdl/bigdl-lora-finetuing-entrypoint.sh
```

After minutes, it is expected to get results like:

```
Training Alpaca-LoRA model with params:
...
Related params
...
world_size: 2!!
PMI_RANK(local_rank): 1
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.28s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:05<00:00,  2.62s/it]
trainable params: 4194304 || all params: 6742609920 || trainable%: 0.06220594176090199
[INFO] spliting and shuffling dataset...
[INFO] shuffling and tokenizing train data...
Map:   2%|▏         | 1095/49759 [00:00<00:30, 1599.00 examples/s]trainable params: 4194304 || all params: 6742609920 || trainable%: 0.06220594176090199
[INFO] spliting and shuffling dataset...
[INFO] shuffling and tokenizing train data...
Map: 100%|██████████| 49759/49759 [00:29<00:00, 1678.89 examples/s]
[INFO] shuffling and tokenizing test data...
Map: 100%|██████████| 49759/49759 [00:29<00:00, 1685.42 examples/s]
[INFO] shuffling and tokenizing test data...
Map: 100%|██████████| 2000/2000 [00:01<00:00, 1573.61 examples/s]
Map: 100%|██████████| 2000/2000 [00:01<00:00, 1578.71 examples/s]
[INFO] begining the training of transformers...
[INFO] Process rank: 0, device: cpudistributed training: True
  0%|          | 1/1164 [02:42<52:28:24, 162.43s/it]
```

You can run BF16-Optimized lora finetuning on kubernetes with OneCCL. So for kubernetes users, please refer to [here](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/finetune/lora/cpu#run-bf16-optimized-lora-finetuning-on-kubernetes-with-oneccl).

## BigDL LLM Fine Tuning on XPU

The following shows how to fine-tune LLM with Quantization (QLoRA built on BigDL-LLM 4bit optimizations) in a docker environment, which is accelerated by Intel XPU.

### 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/bigdl-llm-finetune-qlora-xpu:2.5.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-qlora-xpu:2.5.0-SNAPSHOT \
  -f ./Dockerfile .
```

### 2. Prepare Base Model, Data and Container

Here, we try to fine-tune a [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [English Quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset, and please download them and start a docker container with files mounted like below:

```bash
export BASE_MODE_PATH=your_downloaded_base_model_path
export DATA_PATH=your_downloaded_data_path
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=bigdl-llm-fintune-qlora-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/english_quotes \
   --shm-size="16g" \
   intelanalytics/bigdl-llm-fintune-qlora-xpu:2.5.0-SNAPSHOT
```

The download and mount of base model and data to a docker container demonstrates a standard fine-tuning process. You can skip this step for a quick start, and in this way, the fine-tuning codes will automatically download the needed files:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=bigdl-llm-fintune-qlora-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   --shm-size="16g" \
   intelanalytics/bigdl-llm-fintune-qlora-xpu:2.5.0-SNAPSHOT
```

However, we do recommend you to handle them manually, because the automatical download can be blocked by Internet access and Huggingface authentication etc. according to different environment, and the manual method allows you to fine-tune in a custom way (with different base model and dataset).

### 3. Start Fine-Tuning

Enter the running container:

```bash
docker exec -it bigdl-llm-fintune-qlora-xpu bash
```

Then, start QLoRA fine-tuning:

```bash
bash start-qlora-finetuning-on-xpu.sh
```

After minutes, it is expected to get results like:

```bash
{'loss': 2.256, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.8869, 'learning_rate': 0.00017777777777777779, 'epoch': 0.06}
{'loss': 1.5334, 'learning_rate': 0.00015555555555555556, 'epoch': 0.1}
{'loss': 1.4975, 'learning_rate': 0.00013333333333333334, 'epoch': 0.13}
{'loss': 1.3245, 'learning_rate': 0.00011111111111111112, 'epoch': 0.16}
{'loss': 1.2622, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.19}
{'loss': 1.3944, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.22}
{'loss': 1.2481, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.26}
{'loss': 1.3442, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.29}
{'loss': 1.3256, 'learning_rate': 0.0, 'epoch': 0.32}
{'train_runtime': 204.4633, 'train_samples_per_second': 3.913, 'train_steps_per_second': 0.978, 'train_loss': 1.5072882556915284, 'epoch': 0.32}
100%|██████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:24<00:00,  1.02s/it]
TrainOutput(global_step=200, training_loss=1.5072882556915284, metrics={'train_runtime': 204.4633, 'train_samples_per_second': 3.913, 'train_steps_per_second': 0.978, 'train_loss': 1.5072882556915284, 'epoch': 0.32})
```
