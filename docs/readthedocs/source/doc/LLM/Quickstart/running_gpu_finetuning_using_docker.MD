## BigDL-LLM finetuning on XPU quick start

This quickstart guide walks you through setting up and running large language model finetuning with `bigdl-llm` using docker. 

### Prepare Docker Image

You can download directly from Dockerhub like（recommended):

```bash
docker pull intelanalytics/bigdl-llm-finetune-qlora-xpu:2.5.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone
```

Or follow steps provided in [Build/Use BigDL-LLM xpu image](https://github.com/intel-analytics/BigDL/blob/main/docker/llm/README.md#docker-installation-guide-for-bigdl-llm-fine-tuning-on-xpu) to build the image from source:
```bash
# For standalone
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-qlora-xpu:2.5.0-SNAPSHOT \
  -f ./Dockerfile .
```

Here we use Linux/MacOS as example, if you have a Windows OS, please follow [BigDL-LLM on Windows](https://github.com/intel-analytics/BigDL/blob/main/docker/llm/README.md#bigdl-llm-on-windows) to prepare a BigDL-LLM finetuning image on XPU.

### Prepare Base Model and Container
Here, we try to fine-tune a Llama2-7b with yahma/alpaca-cleaned dataset, and please download them and start a docker container with files mounted like below:
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
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   intelanalytics/bigdl-llm-fintune-qlora-xpu:2.5.0-SNAPSHOT
```

### Start Fine-Tuning (Local Mode)
Enter the running container:

```bash
docker exec -it bigdl-llm-fintune-qlora-xpu bash
```

Then, start QLoRA fine-tuning: If the machine memory is not enough, you can try to set use_gradient_checkpointing=True.

```bash
cd /bigdl
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

### Merge the adapter into the original model
Using the export_merged_model.py to merge.
``` bash
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```
Then you can use ./outputs/checkpoint-200-merged as a normal huggingface transformer model to do inference.
