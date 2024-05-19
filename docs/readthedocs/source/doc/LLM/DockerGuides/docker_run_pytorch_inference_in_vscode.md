# Run/Develop PyTorch in VSCode with Docker on Intel GPU

An IPEX-LLM container is a pre-configured environment that includes all necessary dependencies for running LLMs on Intel GPUs. 

This guide provides steps to run/develop PyTorch examples in VSCode with Docker on Intel GPUs.

```eval_rst
.. note::

   This guide assumes you have already installed VSCode in your environment. To run/develop on Windows, install VSCode and then follow the steps below. To run/develop on Linux, you might open VSCode first and SSH to a remote Linux machine, then proceed with the following steps.

```


## Install Docker

Follow the [Docker installation Guide](./docker_windows_gpu.html#install-docker) to install docker on either Linux or Windows.

## Install Extensions for VSCcode

#### Install Dev Containers Extension
For both Linux/Windows, you will need to Install Dev Container extension.

Open the Extensions view in VSCode (you can use the shortcut Ctrl+Shift+X), then search for and install the "Dev Containers" extension.

<gif>

#### Install WSL Extension for Windows

For Windows, you will need to install wsl extension to to the WSL environment. Open the Extensions view in VSCode (you can use the shortcut `Ctrl+Shift+X`), then search for and install the "WSL" extension.

Press F1 to bring up the Command Palette and type in "WSL: Connect to WSL Using Distro" and select it and then select a specific WSL distro `Ubuntu`

<video>

## Launch Container

Open the Terminal in VSCode (you can use the shortcut `Ctrl+Shift+&#96;`), then pull ipex-llm-xpu Docker Image:

```bash
docker pull intelanalytics/ipex-llm-xpu:latest
```

Start ipex-llm-xpu Docker Container:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

        export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:latest
        export CONTAINER_NAME=my_container
        export MODEL_PATH=/llm/models[change to your model path]

        docker run -itd \
            --net=host \
            --device=/dev/dri \
            --memory="32G" \
            --name=$CONTAINER_NAME \
            --shm-size="16g" \
            -v $MODEL_PATH:/llm/models \
            $DOCKER_IMAGE

   .. tab:: Windows WSL

      .. code-block:: bash

         #/bin/bash
        export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:latest
        export CONTAINER_NAME=my_container
        export MODEL_PATH=/llm/models[change to your model path]

        sudo docker run -itd \
                --net=host \
                --privileged \
                --device /dev/dri \
                --memory="32G" \
                --name=$CONTAINER_NAME \
                --shm-size="16g" \
                -v $MODEL_PATH:/llm/llm-models \
                -v /usr/lib/wsl:/usr/lib/wsl \ 
                $DOCKER_IMAGE
```

Press F1 to bring up the Command Palette and type in "Dev Containers: Attach to Running Container..." and select it and then select `my_container`


## Run/Develop Pytorch Examples

Now you are in a running Docker Container, Open folder `/ipex-llm/python/llm/example/GPU/HF-Transformers-AutoModels/Model/`.

In this folder, we provide several PyTorch examples that you could apply IPEX-LLM INT4 optimizations on models on Intel GPUs.

<video>

For example, if your model is Llama-2-7b-chat-hf and mounted on /llm/models, you can navigate to llama2 directory, excute the following command to run example:
  ```bash
  cd <model_dir>
  python ./generate.py --repo-id-or-model-path /llm/models/Llama-2-7b-chat-hf --prompt PROMPT --n-predict N_PREDICT
  ```


Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

**Sample Output**
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<s>[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  Artificial intelligence (AI) is the broader field of research and development aimed at creating machines that can perform tasks that typically require human intelligence,
```

You can develop your own PyTorch example based on these examples.