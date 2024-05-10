# Run Performance Benchmarking in Docker with IPEX-LLM on Intel GPU

Benchmarking IPEX-LLM on Intel GPUs within Docker can be efficiently achieved using provided benchmark scripts. Follow these steps to execute the process smoothly.

## Install Docker

1. Linux Installation

    Follow the instructions in this [guide](https://www.docker.com/get-started/) to install Docker on Linux.

2. Windows Installation

    For Windows installation, refer to this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/docker_windows_gpu.html#install-docker-on-windows).

## Prepare ipex-llm-xpu Docker Image

Run the following command to pull image from dockerhub:
```bash
docker pull intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```

## Quick Performance Benchmark

Execute a quick performance benchmark by starting the ipex-llm-xpu container, specifying the model, test API, and device, then running the benchmark.sh script. 

### Start ipex-llm-xpu Docker Container
To map the XPU into the container, specify `--device=/dev/dri` when booting the container.
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models [change to your model path]

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models \
        -e REPO_IDS="meta-llama/Llama-2-7b-chat-hf" \
        -e TEST_APIS="transformer_int4_gpu" \
        -e DEVICE=Arc \
        $DOCKER_IMAGE /llm/benchmark.sh
```

Customize environment variables to specify:

- **REPO_IDS:** Model's name and organization, separated by commas if multiple values exist.
- **TEST_APIS:** Different test functions based on the machine, separated by commas if multiple values exist.
- **DEVICE:** Type of device - Max, Flex, Arc.

### Result

Upon completion, you can obtain a CSV result file, the content of CSV results will be printed out. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results.


## Running Performance Benchmark with Detailed Configurations

For more detailed configurations, you should first create the container, modify the config.yaml, then run the benchmarking scripts to get the results.

### Start ipex-llm-xpu Docker Container
```bash
export DOCKER_IMAGE=10.239.45.10/arda/intelanalytics/ipex-llm-xpu:test
export CONTAINER_NAME=my_container
export MODEL_PATH=/mnt/disk1/models

docker run -itd \
    --net=host \
    --device=/dev/dri \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    --shm-size="16g" \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE

docker exec -it $CONTAINER_NAME bash
```

### Modify config.yaml

Navigate to benchmark directory, and modify the `config.yaml` under the `all-in-one` folder for benchmark configurations.
```bash
cd /benchmark/all-in-one
vim config.yaml
```

#### config.yaml

```yaml
repo_id:
  # - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
  # - 'liuhaotian/llava-v1.5-7b' # requires a LLAVA_REPO_DIR env variables pointing to the llava dir; added only for gpu win related test_api now
local_model_hub: 'path to your local model hub'
warm_up: 1 # must set >=2 when run "pipeline_parallel_gpu" test_api
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  - "transformer_int4_gpu_win"            # on Intel GPU for Windows
  # - "transformer_int4_fp16_gpu"           # on Intel GPU, use fp16 for non-linear layer (dtype=fp16)
  # - "transformer_int4_fp16_gpu_win"       # on Intel GPU for Windows, use fp16 for non-linear layer (dtype=fp16)
  # - "transformer_int4_loadlowbit_gpu_win" # on Intel GPU for Windows, use load_low_bit API. Please make sure you have used the save.py to save the converted low bit model
  # - "ipex_fp16_gpu"                       # on Intel GPU, use native transformers' AutoModelForCausalLM, dtype=fp16
  # - "bigdl_fp16_gpu"                      # on Intel GPU, use ipex-llm transformers' AutoModelForCausalLM, dtype=fp16, qtype=fp16
  # - "optimize_model_gpu"                  # on Intel GPU, can load any pytorch models include transformer model
  # - "deepspeed_optimize_model_gpu"        # on Intel GPU, deepspeed autotp inference, weight
  # - "pipeline_parallel_gpu"               # on Intel GPU, pipeline parallel inference, layer
  # - "speculative_gpu"                     # on Intel GPU, inference with self-speculative decoding. CPU/MAX  需要显存大
  # - "transformer_int4"                    # on Intel CPU
  # - "native_int4"                         # on Intel CPU
  # - "optimize_model"                      # on Intel CPU
  # - "pytorch_autocast_bf16"               # on Intel CPU
  # - "transformer_autocast_bf16"           # on Intel CPU
  # - "bigdl_ipex_bf16"                     # on Intel CPU, load model in bf16, which convert the relevant layers in the model into BF16 format (same as transformer_int4(low_bit='bf16'))
  # - "bigdl_ipex_int4"                     # on Intel CPU, load model in int4, which convert the relevant layers in the model into INT4 format (same as transformer_int4(low_bit='sym_int4'))
  # - "bigdl_ipex_int8"                     # on Intel CPU, load model in int8, which convert the relevant layers in the model into INT8 format (same as transformer_int4(low_bit='sym_int8'))
  # - "speculative_cpu"                     # on Intel CPU, inference with self-speculative decoding.
  # - "deepspeed_transformer_int4_cpu"      # on Intel CPU, deepspeed autotp inference, weight
cpu_embedding: False # whether put embedding to CPU
streaming: False # whether output in streaming way (only avaiable now for gpu win related test_api)
use_fp16_torch_dtype: True # whether use fp16 for non-linear layer (only avaiable now for "pipeline_parallel_gpu" test_api)
n_gpu: 2 # number of GPUs to use (only avaiable now for "pipeline_parallel_gpu" test_api)
```

Some parameters in the yaml file that you can configure:


- `repo_id`: The name of the model and its organization.
- `local_model_hub`: The folder path where the models are stored on your machine.
- `warm_up`: The number of warmup trials before performance benchmarking (must set to >= 2 when using "pipeline_parallel_gpu" test_api).
- `num_trials`: The number of runs for performance benchmarking (the final result is the average of all trials).
- `low_bit`: The low_bit precision you want to convert to for benchmarking.
- `batch_size`: The number of samples on which the models make predictions in one forward pass.
- `in_out_pairs`: Input sequence length and output sequence length combined by '-'.
- `test_api`: Different test functions for different machines.
- `cpu_embedding`: Whether to put embedding on CPU (only available for windows GPU-related test_api).
- `streaming`: Whether to output in a streaming way (only available for GPU Windows-related test_api).
- `use_fp16_torch_dtype`: Whether to use fp16 for the non-linear layer (only available for "pipeline_parallel_gpu" test_api).
- `n_gpu`: Number of GPUs to use (only available for "pipeline_parallel_gpu" test_api).

Remark: If you want to benchmark the performance without warmup, you can set `warm_up: 0` and `num_trials: 1` in `config.yaml`, and run each single model and in_out_pair separately.


After configuring the `config.yaml`, run the following scripts:
```bash
source ipex-llm-init -g --device <value>
python run.py
```


### Result

After the benchmarking is completed, you can obtain a CSV result file under the current folder. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results. You can also check whether the column `actual input/output tokens` is consistent with the column `input/output tokens` and whether the parameters you specified in `config.yaml` have been successfully applied in the benchmarking.

