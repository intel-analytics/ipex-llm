## Fine-tune LLM with One CPU

### 1. Prepare IPEX LLM image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/ipex-llm-finetune-lora-cpu:2.2.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-lora-cpu:2.2.0-SNAPSHOT \
  -f ./Dockerfile .
```

### 2. Prepare Base Model, Data and Container

Here, we try to finetune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [Cleaned alpaca data](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json), which contains all kinds of general knowledge and has already been cleaned. And please download them and start a docker container with files mounted like below:

```
docker run -itd \
 --name=ipex-llm-fintune-lora-cpu \
 --cpuset-cpus="your_expected_range_of_cpu_numbers" \
 -e STANDALONE_DOCKER=TRUE \
 -e WORKER_COUNT_DOCKER=your_worker_count \
 -v your_downloaded_base_model_path:/ipex_llm/model \
 -v your_downloaded_data_path:/ipex_llm/data/alpaca_data_cleaned_archive.json \
 intelanalytics/ipex-llm-finetune-lora-cpu:2.2.0-SNAPSHOT \
 bash
```

You can adjust the configuration according to your own environment. After our testing, we recommend you set worker_count=1, and then allocate 80G memory to Docker.

### 3. Start Finetuning

Enter the running container:

```
docker exec -it ipex-llm-fintune-lora-cpu bash
```

Then, run the script to start finetuning:

```
bash /ipex_llm/ipex-llm-lora-finetuing-entrypoint.sh
```

After minutes, it is expected to get results like:

```
Training Alpaca-LoRA model with params:
base_model: /ipex_llm/model/
data_path: /ipex_llm/data/alpaca_data_cleaned_archive.json
output_dir: /home/mpiuser/finetuned_model
batch_size: 128
micro_batch_size: 8
num_epochs: 3
learning_rate: 0.0003
cutoff_len: 256
val_set_size: 2000
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ['q_proj', 'v_proj']
train_on_inputs: True
group_by_length: False
wandb_project:
wandb_run_name:
wandb_watch:
wandb_log_model:
resume_from_checkpoint: None
use_ipex: False
bf16: False

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
2023:10:11-01:12:20:(  670) |CCL_WARN| no membind support for NUMA node 0, skip thread membind
2023:10:11-01:12:20:(  671) |CCL_WARN| no membind support for NUMA node 1, skip thread membind
2023:10:11-01:12:20:(  672) |CCL_WARN| no membind support for NUMA node 0, skip thread membind
2023:10:11-01:12:20:(  673) |CCL_WARN| no membind support for NUMA node 1, skip thread membind
[INFO] begining the training of transformers...
[INFO] Process rank: 0, device: cpudistributed training: True
  0%|          | 1/1164 [02:42<52:28:24, 162.43s/it]
```
