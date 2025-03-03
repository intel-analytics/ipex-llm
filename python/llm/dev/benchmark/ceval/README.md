## C-Eval Benchmark Test Guide

This guide provides instructions for running the C-Eval benchmark test in both single-GPU and multi-GPU environments. [C-Eval](https://cevalbenchmark.com) is a comprehensive multi-level, multi-discipline Chinese evaluation suite for foundational models. It consists of 13,948 multiple-choice questions spanning 52 diverse disciplines and four difficulty levels. For more details, see the [C-Eval paper](https://arxiv.org/abs/2305.08322) and [GitHub repository](https://github.com/hkust-nlp/ceval).

---

### Single-GPU Environment

#### 1. Download Dataset

Download and unzip the dataset for evaluation:
```bash
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data
mv ceval-exam.zip data
cd data; unzip ceval-exam.zip
```

#### 2. Run Evaluation

Use the following command to run the evaluation:
```bash
bash run.sh
```

Contents of `run.sh`:
```bash
export IPEX_LLM_LAST_LM_HEAD=0
python eval.py \
    --model_path "path to model" \
    --eval_type validation \
    --device xpu \
    --eval_data_path data \
    --qtype sym_int4
```

> **Note**
>
> - `eval_type`: There are two types of evaluations:
>   - `validation`: Runs on the validation dataset and outputs evaluation scores.
>   - `test`: Runs on the test dataset and outputs a `submission.json` file for submission on [C-Eval](https://cevalbenchmark.com) to get evaluation scores.

---

### Multi-GPU Environment

#### 1. Prepare Environment

1. **Set Docker Image and Container Name**:
   ```bash
   export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
   export CONTAINER_NAME=ceval-benchmark
   ```

2. **Start Docker Container**:
   ```bash
   docker run -td \
         --net=host \
         --group-add video \
         --device=/dev/dri \
         --name=$CONTAINER_NAME \
         -v /home/intel/LLM:/llm/models/ \
         -e no_proxy=localhost,127.0.0.1 \
         -e http_proxy=$HTTP_PROXY \
         -e https_proxy=$HTTPS_PROXY \
         --shm-size="16g" \
         $DOCKER_IMAGE
   ```

3. **Enter the Container**:
   ```bash
   docker exec -it $CONTAINER_NAME bash
   ```

#### 2. Configure `lm-evaluation-harness`

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness
   ```

2. **Update Multi-GPU Support File**:
   Update `lm_eval/models/vllm_causallms.py` based on the following link:
   [Update Multi-GPU Support File](https://github.com/EleutherAI/lm-evaluation-harness/compare/main...liu-shaojun:lm-evaluation-harness:multi-arc?expand=1)

3. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

#### 3. Configure Environment Variables

Set environment variables required for multi-GPU execution:
```bash
export CCL_WORKER_COUNT=2
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
export CCL_SAME_STREAM=1
export CCL_BLOCKING_WAIT=0

export SYCL_CACHE_PERSISTENT=1
export FI_PROVIDER=shm
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0
```

Load Intel OneCCL environment variables:
```bash
source /opt/intel/1ccl-wks/setvars.sh
```

#### 4. Run Evaluation

Use the following command to run the C-Eval benchmark:
```bash
lm_eval --model vllm \
  --model_args pretrained=/llm/models/CodeLlama-34b/,dtype=float16,max_model_len=2048,device=xpu,load_in_low_bit=fp8,tensor_parallel_size=4,distributed_executor_backend="ray",gpu_memory_utilization=0.90,trust_remote_code=True \
  --tasks ceval-valid \
  --batch_size 2 \
  --num_fewshot 0 \
  --output_path c-eval-result
```

#### 5. Notes

- **Model and Parameter Adjustments**:
  - **`pretrained`**: Replace with the desired model path, e.g., `/llm/models/CodeLlama-7b/`.
  - **`load_in_low_bit`**: Set to `fp8` or other precision options based on hardware and task requirements.
  - **`tensor_parallel_size`**: Adjust based on the number of GPUs and memory. Recommended to match the GPU count.
  - **`batch_size`**: Increase to accelerate testing, but ensure it does not cause OOM errors. Recommended values are `2` or `3`.
  - **`num_fewshot`**: Specify the number of few-shot examples. Default is `0`. Increasing this value can improve model contextual understanding but may significantly increase input length and runtime.

- **Logging**:
  To log both to the console and a file, use:
  ```bash
  lm_eval --model vllm ... | tee c-eval.log
  ```

- **Container Debugging**:
  Ensure the paths for the model and tasks are correctly set, e.g., check if `/llm/models/` is properly mounted in the container.

---

By following the above steps, you can successfully run the C-Eval benchmark in both single-GPU and multi-GPU environments.

