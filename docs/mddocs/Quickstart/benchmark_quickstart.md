# Run Performance Benchmarking with IPEX-LLM

We can perform benchmarking for IPEX-LLM on Intel CPUs and GPUs using the benchmark scripts we provide.

## Table of Contents
- [Prepare the Environment](./benchmark_quickstart.md#prepare-the-environment)
- [Prepare the Scripts](./benchmark_quickstart.md#prepare-the-scripts)
- [Run on Windows](./benchmark_quickstart.md#run-on-windows)
- [Run on Linux](./benchmark_quickstart.md#run-on-linux)
- [Result](./benchmark_quickstart.md#result)

## Prepare the Environment

You can refer to [here](../Overview/install.md) to install IPEX-LLM in your environment. The following dependencies are also needed to run the benchmark scripts.

```
pip install pandas
pip install omegaconf
```

## Prepare the Scripts

Navigate to your local workspace and then download IPEX-LLM from GitHub. Modify the `config.yaml` under `all-in-one` folder for your benchmark configurations.

```
cd your/local/workspace
git clone https://github.com/intel-analytics/ipex-llm.git
cd ipex-llm/python/llm/dev/benchmark/all-in-one/
```

### config.yaml


```yaml
repo_id:
  - 'meta-llama/Llama-2-7b-chat-hf'
local_model_hub: 'path to your local model hub'
warm_up: 1 # must set >=2 when run "pipeline_parallel_gpu" test_api
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
  - '2048-256'
test_api:
  - "transformer_int4_gpu"   # on Intel GPU, transformer-like API, (qtype=int4)
cpu_embedding: False # whether put embedding to CPU
streaming: False # whether output in streaming way (only avaiable now for gpu win related test_api)
task: 'continuation' # task can be 'continuation', 'QA' and 'summarize'
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
  - `transformer_int4_gpu` on Intel GPU for Linux
  - `transformer_int4_gpu_win` on Intel GPU for Windows
  - `transformer_int4` on Intel CPU
- `cpu_embedding`: Whether to put embedding on CPU (only available for windows GPU-related test_api).
- `streaming`: Whether to output in a streaming way (only available for GPU Windows-related test_api).
- `use_fp16_torch_dtype`: Whether to use fp16 for the non-linear layer (only available for "pipeline_parallel_gpu" test_api).
- `n_gpu`: Number of GPUs to use (only available for "pipeline_parallel_gpu" test_api).
- `task`: There are three tasks: `continuation`, `QA` and `summarize`. `continuation` refers to writing additional content based on prompt. `QA` refers to answering questions based on prompt. `summarize` refers to summarizing the prompt.


> [!NOTE]
> If you want to benchmark the performance without warmup, you can set ``warm_up: 0`` and ``num_trials: 1`` in ``config.yaml``, and run each single model and in_out_pair separately. 


## Run on Windows

Please refer to [here](../Overview/install_gpu.md#runtime-configuration) to configure oneAPI environment variables. Choose corresponding commands base on your device.

- For **Intel iGPU** and **Intel Arc™ A-Series Graphics**:

  ```bash
  set SYCL_CACHE_PERSISTENT=1
  
  python run.py
  ```

## Run on Linux

Please choose corresponding commands base on your device.

- For **Intel Arc™ A-Series** and **Intel Data Center GPU Flex**:

  For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:
  
  ```bash
  ./run-arc.sh
  ```

- For **Intel iGPU**:

  For Intel iGPU, we recommend:
  
  ```bash
  ./run-igpu.sh
  ```

- For **Intel Data Center GPU Max**:

  Please note that you need to run ``conda install -c conda-forge -y gperftools=2.10`` before running the benchmark script on Intel Data Center GPU Max Series.
  
  ```bash
  ./run-max-gpu.sh
  ```

- For **Intel SPR**:

  For Intel SPR machine, we recommend:
  
  ```bash
  ./run-spr.sh
  ```

  The scipt uses a default numactl strategy. If you want to customize it, please use ``lscpu`` or ``numactl -H`` to check how cpu indexs are assigned to numa node, and make sure the run command is binded to only one socket.

- For **Intel HBM**:

  For Intel HBM machine, we recommend:
  
  ```bash
  ./run-hbm.sh
  ```
  
  The scipt uses a default numactl strategy. If you want to customize it, please use ``numactl -H`` to check how the index of hbm node and cpu are assigned.

  For example:
  
  ```bash
  node   0   1   2   3
     0:  10  21  13  23
     1:  21  10  23  13
     2:  13  23  10  23
     3:  23  13  23  10
  ```
  
  here hbm node is the node whose distance from the checked node is 13, node 2 is node 0's hbm node.
  
  And make sure the run command is binded to only one socket.

## Result

After the benchmarking is completed, you can obtain a CSV result file under the current folder. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results. You can also check whether the column `actual input/output tokens` is consistent with the column `input/output tokens` and whether the parameters you specified in `config.yaml` have been successfully applied in the benchmarking.
