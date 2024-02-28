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

Here, we try to fine-tune a [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) with [English Quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset, and please download them and start a docker container with files mounted like below:

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
   -v $DATA_PATH:/bigdl/data/english_quotes \
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
{'loss': 2.256, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.8869, 'learning_rate': 0.00017777777777777779, 'epoch': 0.06}
{'loss': 1.5334, 'learning_rate': 0.00015555555555555556, 'epoch': 0.1}
{'loss': 1.4975, 'learning_rate': 0.00013333333333333334, 'epoch': 0.13}
{'loss': 1.3245, 'learning_rate': 0.00011111111111111112, 'epoch': 0.16}
{'loss': 1.2622, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.19}
{'loss': 1.3944, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.22}
{'loss': 1.2481, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.26}
{'loss': 1.3442, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.29}
{'loss': 1.3256, 'learning_rate': 0.0, 'epoch': 0.32}
{'train_runtime': xxx, 'train_samples_per_second': xxx, 'train_steps_per_second': xxx, 'train_loss': 1.5072882556915284, 'epoch': 0.32}
100%|██████████████████████████████████████████████████████████████████████████████████████| 200/200 [xx:xx<xx:xx,  xxxs/it]
TrainOutput(global_step=200, training_loss=1.5072882556915284, metrics={'train_runtime': xxx, 'train_samples_per_second': xxx, 'train_steps_per_second': xxx, 'train_loss': 1.5072882556915284, 'epoch': 0.32})
```

### 4. Merge the adapter into the original model

Using the [export_merged_model.py](../../../../../../python/llm/example/GPU/LLM-Finetuning/QLoRA/export_merged_model.py) to merge.

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.

### 5. Use BigDL-LLM to verify the fine-tuning effect

Train more steps and try input sentence like `['quote'] -> [?]` to verify. For example, using `“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->: ` to inference.
BigDL-LLM llama2 example [link](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2). Update the `LLAMA2_PROMPT_FORMAT = "{prompt}"`.

```bash
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt "“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->:"  --n-predict 20
```

#### Sample Output

Base_model output

```log
Inference time: xxx s
-------------------- Prompt --------------------
“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->:
-------------------- Output --------------------
“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->: 💻 Fine-tuning a language model on a powerful device like an Intel CPU
```

Merged_model output

```log
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Inference time: xxx s
-------------------- Prompt --------------------
“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->:
-------------------- Output --------------------
“QLoRA fine-tuning using BigDL-LLM 4bit optimizations on Intel CPU is Efficient and convenient” ->: ['bigdl'] ['deep-learning'] ['distributed-computing'] ['intel'] ['optimization'] ['training'] ['training-speed']
```

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
