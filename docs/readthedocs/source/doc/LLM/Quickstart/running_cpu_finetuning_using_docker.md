## IPEX-LLM finetuning on CPU quick start

This quickstart guide walks you through setting up and running large language model finetuning with `ipex-llm` using a docker image. 

### Prepare Docker Image

You can download directly from Dockerhub :

```bash
docker pull intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.1.0-SNAPSHOT
```


### Prepare Base Model and dataset
Here, we fine-tune a Llama2-7b with yahma/alpaca-cleaned dataset, please download them first.

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
   --name=ipex-llm-fintune-qlora-cpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/ipex_llm/model \
   -v $DATA_PATH:/ipex_llm/data/alpaca-cleaned \
   intelanalytics/ipex-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT
```

### Start Fine-Tuning (Local Mode)
Enter the running container:

```bash
docker exec -it ipex-llm-fintune-qlora-cpu bash
```

Then, start QLoRA fine-tuning: If the machine memory is not enough, you can try to set use_gradient_checkpointing=True.

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