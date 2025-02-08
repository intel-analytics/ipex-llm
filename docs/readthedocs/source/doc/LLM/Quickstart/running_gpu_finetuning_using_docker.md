## IPEX-LLM inference on GPU quick start

This quickstart guide walks you through setting up and running large language model inference with `ipex-llm` using docker. 

### Docker Installation Instructions
**For New Users**:
- Begin by visiting the official Docker Get Started page for a comprehensive introduction and installation guide.

### Prepare Docker Image

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT
```

### Prepare Base Model and dataset
Here, we fine-tune a Llama2-7b with yahma/alpaca-cleaned dataset, please follow the steps to download them first.

#### Download base model
[Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is used as example to show LLM finetuning. Create a ``download.py`` and insert the code snippet below to dwonload the model from huggingface. 

``` python
from huggingface_hub import snapshot_download
repo_id="meta-llama/Llama-2-7b-chat-hf"
local_dir="/home/llm/models/Llama-2-7b-chat-hf"
snapshot_download(repo_id=repo_id,
                  local_dir=local_dir,
                  local_dir_use_symlinks=False
                  )
```

Then use the bash script to download the model to local directory of ``/home/llm/models/Llama-2-7b-chat-hf``. 

``` bash
pip install huggingface_hub
python download.py
```

#### Download yahma/alpaca-cleaned dataset
``` bash
mkdir alpaca-cleaned
cd alpaca-cleaned
```
Go to huggingface website https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main, click ``Download file`` of ``alpaca_data_cleaned.json`` and save it to ``aplace-cleaned``.

### Start a docker container
Once you have base model and dataset ready, you can start a docker container with files mounted correctly:

```bash
export BASE_MODE_PATH=/home/llm/models/Llama-2-7b-chat-hf
export DATA_PATH=./alpaca-cleaned
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --privileged \
   --name=ipex-llm-fintune-qlora-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   intelanalytics/ipex-llm-finetune-qlora-xpu:2.1.0-SNAPSHOT
```

### Start Fine-Tuning
Enter the running container:

```bash
docker exec -it ipex-llm-fintune-qlora-xpu bash
```

Then, start QLoRA fine-tuning: If the machine memory is not enough, you can try to set use_gradient_checkpointing=True.

```bash
bash start-qlora-finetuning-on-xpu.sh
```

After minutes, it is expected to get results like:
```bash 
{'loss': 1.8549, 'learning_rate': 2e-05, 'epoch': 0.02}
{'loss': 1.7911, 'learning_rate': 1.7777777777777777e-05, 'epoch': 0.03}
{'loss': 1.5979, 'learning_rate': 1.555555555555556e-05, 'epoch': 0.05}
{'loss': 1.4751, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.06}
{'loss': 1.3425, 'learning_rate': 1.1111111111111113e-05, 'epoch': 0.08}
{'loss': 1.2782, 'learning_rate': 8.888888888888888e-06, 'epoch': 0.09}
{'loss': 1.2633, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.11}
{'loss': 1.1979, 'learning_rate': 4.444444444444444e-06, 'epoch': 0.12}
{'loss': 1.1929, 'learning_rate': 2.222222222222222e-06, 'epoch': 0.14}
{'loss': 1.2319, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': 130.4622, 'train_samples_per_second': 6.132, 'train_steps_per_second': 1.533, 'train_loss': 1.4225698184967042, 'epoch': 0.15}
100%|██████████████████████████████████████████████████████████████████████████| 200/200 [02:10<00:00,  1.53it/s]
TrainOutput(global_step=200, training_loss=1.4225698184967042, metrics={'train_runtime': 130.4622, 'train_samples_per_second': 6.132, 'train_steps_per_second': 1.533, 'train_loss': 1.4225698184967042, 'epoch': 0.15})
```

### Troubleshooting
Please refer to [git repo](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/README.md#troubleshooting) for solutions of common issues during finetuning.