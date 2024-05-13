# Run PyTorch Inference on Intel GPU using Docker (on Linux or WSL)

We can run PyTorch Inference Benchmark, Chat Service and PyTorch Examples on Intel GPUs within Docker (on Linux or WSL).

## Install Docker

1. Linux Installation

    Follow the instructions in this [guide](https://www.docker.com/get-started/) to install Docker on Linux.

2. Windows Installation

    For Windows installation, refer to this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/docker_windows_gpu.html#install-docker-on-windows).

## Launch Docker

Prepare ipex-llm-xpu Docker Image:
```bash
docker pull intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```

Start ipex-llm-xpu Docker Container:
```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

docker run -itd \
    --net=host \
    --device=/dev/dri \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    --shm-size="16g" \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE
```

Access the container:
```
docker exec -it $CONTAINER_NAME bash
```

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

```eval_rst
.. tip::

  You can run the Env-Check script to verify your ipex-llm installation and runtime environment.
  .. code-block:: bash

      bash env-check.sh

```

## Run Inference Benchmark 

Navigate to benchmark directory, and modify the `config.yaml` under the `all-in-one` folder for benchmark configurations.
```bash
cd /benchmark/all-in-one
vim config.yaml
```

**Modify config.yaml**
```eval_rst
.. note::

  ``dtype``: The model is originally loaded in this data type.  After ipex-llm conversion, all the non-linear layers remain to use this data type.

  ``qtype``: ipex-llm will convert all the linear-layers' weight to this data type.
```


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
  - "transformer_int4_gpu"                # on Intel GPU, transformer-like API, (qtype=int4)
  # - "transformer_int4_gpu_win"            # on Intel GPU for Windows, transformer-like API, (qtype=int4)
  # - "transformer_int4_fp16_gpu"           # on Intel GPU, transformer-like API, (qtype=int4), (dtype=fp16)
  # - "transformer_int4_fp16_gpu_win"       # on Intel GPU for Windows, transformer-like API, (qtype=int4), (dtype=fp16)
  # - "transformer_int4_loadlowbit_gpu_win" # on Intel GPU for Windows, transformer-like API, (qtype=int4), use load_low_bit API. Please make sure you have used the save.py to save the converted low bit model
  # - "ipex_fp16_gpu"                       # on Intel GPU, use native transformers API, (dtype=fp16)
  # - "bigdl_fp16_gpu"                      # on Intel GPU, use ipex-llm transformers API, (dtype=fp16), (qtype=fp16)
  # - "optimize_model_gpu"                  # on Intel GPU, can optimize any pytorch models include transformer model
  # - "deepspeed_optimize_model_gpu"        # on Intel GPU, deepspeed autotp inference
  # - "pipeline_parallel_gpu"               # on Intel GPU, pipeline parallel inference
  # - "speculative_gpu"                     # on Intel GPU, inference with self-speculative decoding
  # - "transformer_int4"                    # on Intel CPU, transformer-like API, (qtype=int4)
  # - "native_int4"                         # on Intel CPU
  # - "optimize_model"                      # on Intel CPU, can optimize any pytorch models include transformer model
  # - "pytorch_autocast_bf16"               # on Intel CPU
  # - "transformer_autocast_bf16"           # on Intel CPU
  # - "bigdl_ipex_bf16"                     # on Intel CPU, (qtype=bf16)
  # - "bigdl_ipex_int4"                     # on Intel CPU, (qtype=int4)
  # - "bigdl_ipex_int8"                     # on Intel CPU, (qtype=int8)
  # - "speculative_cpu"                     # on Intel CPU, inference with self-speculative decoding
  # - "deepspeed_transformer_int4_cpu"      # on Intel CPU, deepspeed autotp inference
cpu_embedding: False # whether put embedding to CPU
streaming: False # whether output in streaming way (only avaiable now for gpu win related test_api)
use_fp16_torch_dtype: True # whether use fp16 for non-linear layer (only avaiable now for "pipeline_parallel_gpu" test_api)
n_gpu: 2 # number of GPUs to use (only avaiable now for "pipeline_parallel_gpu" test_api)
```

Some parameters in the yaml file that you can configure:


- `repo_id`: The name of the model and its organization.
- `local_model_hub`: The folder path where the models are stored on your machine. Replace 'path to your local model hub' with /llm/models.
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


```eval_rst
.. note::

  If you want to benchmark the performance without warmup, you can set ``warm_up: 0`` and ``num_trials: 1`` in ``config.yaml``, and run each single model and in_out_pair separately. 
```


After configuring the `config.yaml`, run the following scripts:
```bash
source ipex-llm-init --gpu --device <value>
python run.py
```


**Result**

After the benchmarking is completed, you can obtain a CSV result file under the current folder. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results. You can also check whether the column `actual input/output tokens` is consistent with the column `input/output tokens` and whether the parameters you specified in `config.yaml` have been successfully applied in the benchmarking.


## Run Chat Service

We provide `chat.py` for conversational AI. 

For example, if your model is Llama-2-7b-chat-hf and mounted on /llm/models, you can execute the following command to initiate a conversation:
  ```bash
  cd /llm
  python chat.py --model-path /llm/models/Llama-2-7b-chat-hf
  ```

Here is a demostration:

<a align="left"  href="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-chatpy-demo.gif">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llm-inference-cpu-docker-chatpy-demo.gif" width='60%' /> 

</a><br>

## Run PyTorch Examples

We provide several PyTorch examples that you could apply IPEX-LLM INT4 optimizations on models on Intel GPUs

For example, if your model is Llama-2-7b-chat-hf and mounted on /llm/models, you can navigate to /examples/llama2 directory, excute the following command to run example:
  ```bash
  cd /examples/<model_dir>
  python ./generate.py --repo-id-or-model-path /llm/models/Llama-2-7b-chat-hf --prompt PROMPT --n-predict N_PREDICT
  ```


Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

**Sample Output**
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<s>[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  Artificial intelligence (AI) is the broader field of research and development aimed at creating machines that can perform tasks that typically require human intelligence,
```