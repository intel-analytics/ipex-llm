# Benchmarking

We can do benchmarking on Intel GPUs and CPUs. You can refer to [this](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install.html) to prepare the environment.

## Clone Repo

Navigate to your local workspace and then download BigDL from GitHub.

```
cd your/local/workspace/
git clone https://github.com/intel-analytics/BigDL.git
```

## Configure YAML File

Replace the config.yaml under all-in-one folder with your own YAML file. The format of the content is as follows.

```yaml
repo_id:
  - 'meta-llama/Llama-2-7b-chat-hf'
local_model_hub: '/mnt/disk1/models'
warm_up: 1
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
  - '2048-256'
test_api:
  - "transformer_int4_gpu"  # on Intel GPU
cpu_embedding: False # whether put embedding to CPU (only avaiable now for gpu win related test_api)
```

The meanings of the different parameter configurations in the yaml file are shown below.


- repo_id: The repo_id consists of two parts, one is the name of the model producer and the other is the name of the folder that actually holds the model data. 
- local_model_hub: The folder path where the models are stored in your machine.
- warm_up: Drop the results of the first time for better performance.
- num_trials: Take the average of the three test result as the final performance result.
- num_beams: Number of beams for beam search. 1 means no beam search.
- low_bit: Convert the model to which precision you want to test.
- batch_size: The number of samples on which the models makes predictions in one forward pass.
- in_out_pairs: Inputs given to the model and expected outputs. 
- test_api: Use different test functions on different machines.
  - `transformer_int4_gpu` on Intel GPU for Linux
  - `transformer_int4_gpu_win` on Intel GPU for Windows
  - `deepspeed_transformer_int4_cpu` on Intel SPR Server
  - `transformer_int4` on Intel CPU
- cpu_embedding: Whether put embedding to the CPU (only avaiable now for windows gpu related test_api)

## Runtime Configuration

You can refer to [windows runtime configuration](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration) and [linux runtime configuration](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration) to source the needed environment variables for Intel GPU.


## Run Script

Run the script, wait for the model to finish running, and then you can obtain a CSV result file under the current folder.

```
cd your/local/workspace/BigDL/python/llm/dev/benchmark/all-in-one/
python run.py
```