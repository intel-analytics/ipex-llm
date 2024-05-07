## Build/Use IPEX-LLM cpp xpu image

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-cpp-xpu:latest .
```

### Start Image

To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. And change the `/path/to/models` to mount the models.

An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpp-xpu:latest
export CONTAINER_NAME=ipex-llm-cpp-xpu-container
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        -v /path/to/models:/models \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```


After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-cpp-xpu-container /bin/bash
```

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```


### Use the image for running llama.cpp inference with IPEX-LLM on Intel GPU

```bash
# mount models and change the model_path in `start-llama-cpp.sh`
cd ~/scripts/
bash start-llama-cpp.sh
```

Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html) for more details.


### Use the image for running Ollama serving with IPEX-LLM on Intel GPU

Running the ollama on the background, you can see the ollama.log in `/root/ollama/ollama.log`
```bash
cd ~/scripts/
bash start-ollama.sh # ctrl+c to exit
# pull model
cd /root/ollama && ./ollama pull dolphin-phi:latest
```

Use the Curl to Test:
```bash
curl http://localhost:11434/api/generate -d '
{ 
   "model": "dolphin-phi", 
   "prompt": "What is AI?", 
   "stream": false
}'
```

Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html#pull-model) for more details.


### Use the image for running Open WebUI with Intel GPU

Start the ollama and load the model first, then use the open-webui to chat.
```bash
cd ~/scripts/
bash start-open-webui.sh
# INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

For how to log-in or other guide, Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/open_webui_with_ollama_quickstart.html) for more details.
