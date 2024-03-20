## Fine-tune LLM with BigDL LLM Container

The following shows how to fine-tune LLM with Quantization (QLoRA built on BigDL-LLM 4bit optimizations) in a docker environment, which is accelerated by Intel CPU.

### 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
# For standalone
docker pull intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT

# For k8s
docker pull intelanalytics/bigdl-llm-finetune-qlora-cpu-k8s:2.5.0-SNAPSHOT
```

Or build the image from source:

```bash
# For standalone
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT \
  -f ./Dockerfile .

# For k8s
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-qlora-cpu-k8s:2.5.0-SNAPSHOT \
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
   --name=bigdl-llm-fintune-qlora-cpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/bigdl/model \
   -v $DATA_PATH:/bigdl/data/alpaca-cleaned \
   intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT
```

The download and mount of base model and data to a docker container demonstrates a standard fine-tuning process. You can skip this step for a quick start, and in this way, the fine-tuning codes will automatically download the needed files:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker run -itd \
   --net=host \
   --name=bigdl-llm-fintune-qlora-cpu \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT
```

However, we do recommend you to handle them manually, because the automatical download can be blocked by Internet access and Huggingface authentication etc. according to different environment, and the manual method allows you to fine-tune in a custom way (with different base model and dataset).

### 3. Start Fine-Tuning (Local Mode)

Enter the running container:

```bash
docker exec -it bigdl-llm-fintune-qlora-cpu bash
```

Then, start QLoRA fine-tuning:
If the machine memory is not enough, you can try to set `use_gradient_checkpointing=True`.

```bash
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

### 4. Start Multi-Porcess Fine-Tuning in One Docker

<img src="https://github.com/Uxito-Ada/BigDL/assets/60865256/f25c43b3-2b24-4476-a0fe-804c0ef3c36c" height="240px"><br>

Multi-process parallelism enables higher performance for QLoRA fine-tuning, e.g. Xeon server series with multi-processor-socket architecture is suitable to run one instance on each QLoRA. This can be done by simply invoke >=2 OneCCL instances in BigDL QLoRA docker:

```bash
docker run -itd \
 --name=bigdl-llm-fintune-qlora-cpu \
 --cpuset-cpus="your_expected_range_of_cpu_numbers" \
 -e STANDALONE_DOCKER=TRUE \
 -e WORKER_COUNT_DOCKER=your_worker_count \
 -v your_downloaded_base_model_path:/bigdl/model \
 -v your_downloaded_data_path:/bigdl/data/alpaca_data_cleaned_archive.json \
 intelanalytics/bigdl-llm-finetune-qlora-cpu-standalone:2.5.0-SNAPSHOT
```

Note that `STANDALONE_DOCKER` is set to **TRUE** here.

Then following the same way as above to enter the docker container and start fine-tuning:

```bash
bash start-qlora-finetuning-on-cpu.sh
```

### 5. Start Distributed Fine-Tuning on Kubernetes

Besides multi-process mode, you can also run QLoRA on a kubernetes cluster. please refer [here](https://github.com/intel-analytics/BigDL/blob/main/docker/llm/finetune/qlora/cpu/kubernetes/README.md).
