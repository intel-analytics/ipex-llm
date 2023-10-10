## Fine-tune LLM with One CPU

### 1. Prepare BigDL image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/bigdl-llm-finetune-lora-cpu:2.4.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-lora-cpu:2.4.0-SNAPSHOT \
  -f ./Dockerfile .
```

### 2. Prepare Base Model, Data and Container

Here, we try to finetune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [Cleaned alpaca data](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json), which contains all kinds of general knowledge and has already been cleaned. And please download them and start a docker container with files mounted like below:

```
docker run -itd \
 --name=bigdl-llm-fintune-lora-cpu \
 -e STANDALONE_DOCKER=TRUE \
 -v your_downloaded_base_model_path:/ppml/model \
 -v your_downloaded_data_path:/ppml/data/alpaca_data_cleaned_archive.json \
 10.239.45.10/arda/intelanalytics/bigdl-llm-finetune-cpu:2.4.0-SNAPSHOT \
 bash
```

### 3. Start Finetuning

Enter the running container:

```
docker exec -it bigdl-llm-fintune-lora-cpu bash
```

Then, run the script to start finetuning:

```
bash /ppml/bigdl-lora-finetuing-entrypoint.sh
```
