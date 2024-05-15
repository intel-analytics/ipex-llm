# Finetune LLM with IPEX LLM Container

The following shows how to finetune LLM with IPEX-LLM optimizations in a docker environment, which is accelerated by Intel XPU.


With this docker image, we can use all [ipex-llm finetune examples on Intel GPU](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning), including:

- [LoRA](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/LoRA): examples of running LoRA finetuning
- [QLoRA](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QLoRA): examples of running QLoRA finetuning
- [QA-LoRA](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA): examples of running QA-LoRA finetuning
- [ReLora](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/ReLora): examples of running ReLora finetuning
- [DPO](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/DPO): examples of running DPO finetuning
- [HF-PEFT](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/HF-PEFT): run finetuning on Intel GPU using Hugging Face PEFT code without modification
- [axolotl](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/axolotl): LLM finetuning on Intel GPU using axolotl without writing code


## 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/ipex-llm-finetune-xpu:2.1.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-xpu:2.1.0-SNAPSHOT \
  -f ./Dockerfile .
```

## 2. Prepare Base Model, Data and Container

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
   --name=ipex-llm-finetune-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   intelanalytics/ipex-llm-finetune-xpu:2.1.0-SNAPSHOT
```

The download and mount of base model and data to a docker container demonstrates a standard fine-tuning process. You can skip this step for a quick start, and in this way, the fine-tuning codes will automatically download the needed files:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=ipex-llm-finetune-xpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   --shm-size="16g" \
   intelanalytics/ipex-llm-finetune-xpu:2.1.0-SNAPSHOT
```

However, we do recommend you to handle them manually, because the download can be blocked by Internet access and Huggingface authentication etc. according to different environment, and the manual method allows you to fine-tune in a custom way (with different base model and dataset).

## 3. Start Fine-Tuning

### 3.1 QLoRA Llama2-7b example

Enter the running container:

```bash
docker exec -it ipex-llm-finetune-xpu bash
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

### 3.2 QA-LoRA Llama2-7b example

Enter the running container:

```bash
docker exec -it ipex-llm-finetune-xpu bash
```

Enter QA-LoRA dir.

```bash
cd /LLM-Finetuning/QA-LoRA
```

Modify configuration in scripts, e.g., `--base_model` and `--data_path` in `qalora_finetune_llama2_7b_arc_1_card.sh`.

Then, start QA-LoRA fine-tuning:

```bash
bash qalora_finetune_llama2_7b_arc_1_card.sh
```

For more details, please refer to [QA-LoRA example](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/QA-LoRA).

### 3.3 Axolotl LoRA

Enter the running container:

```bash
docker exec -it ipex-llm-finetune-xpu bash
```

Enter QA-LoRA dir.

```bash
cd /LLM-Finetuning/axolotl
```

Modify configuration in axolotl config, e.g., `base_model` and `datasets.path` in `lora.yml`.

Then, start QA-LoRA fine-tuning:

```bash
accelerate launch finetune.py lora.yml
```

For more details, please refer to [axolotl example](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning/axolotl).
