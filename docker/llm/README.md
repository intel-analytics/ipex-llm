# IPEX-LLM Docker Containers

You can run IPEX-LLM containers (via docker or k8s) for inference, serving and fine-tuning on Intel CPU and GPU. Details on how to use these containers are available at [IPEX-LLM Docker Container Guides](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Docker/index.html).

### Prerequisites

- Docker on Windows or Linux
- Windows Subsystem for Linux (WSL) is required if using Windows.  

### Quick Start 


#### Pull a IPEX-LLM Docker Image
To pull IPEX-LLM Docker images from [Intel Analytics Docker Hub](https://hub.docker.com/u/intelanalytics), use the `docker pull` command. For instance, to pull the CPU inference image:
```bash
docker pull intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT
```

Available images in hub are: 

| Image Name | Description |
| --- | --- |
| intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT | CPU Inference |
| intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT | GPU Inference |
| intelanalytics/ipex-llm-serving-cpu:2.1.0-SNAPSHOT | CPU Serving|
| intelanalytics/ipex-llm-serving-xpu:2.1.0-SNAPSHOT | GPU Serving|
| intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT | CPU Finetuning via Docker|
|intelanalytics/ipex-llm-finetune-qlora-cpu-k8s:2.1.0-SNAPSHOT|CPU Finetuning via Kubernetes|
| intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT| GPU Finetuning|

#### Run a Container
Use `docker run` command to run an IPEX-LLM docker container. For detailed instructions, refer to the [IPEX-LLM Docker Container Guides](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Docker/index.html).


#### Build Docker Image

To build a Docker image from source, first clone the IPEX-LLM repository and navigate to the Dockerfile directory. For example, to build the CPU inference image, navigate to `docker/llm/inference/cpu/docker`.

Then, use the following command to build the image (replace `your_image_name` with your desired image name):

```bash
docker build \
  --build-arg no_proxy=localhost,127.0.0.1 \
  --rm --no-cache -t your_image_name .
```

> Note: If you're working behind a proxy, also add args `--build-arg http_proxy=http://your_proxy_uri:port` and `--build-arg https_proxy=https://your_proxy_url:port`  
