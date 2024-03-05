# BigDL-LLM Benchmarking

We can do benchmarking on Intel CPUs and GPUs using the benchmark script we provide. You can refer to [this](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install.html) to prepare the environment for bigdl-llm.

## Clone Repo

Navigate to your local workspace and then download BigDL from GitHub. Modify the `config.yaml` under `all-in-one folder` for your own benchmark configurations.

```
git clone https://github.com/intel-analytics/BigDL.git
cd your/local/workspace/BigDL/python/llm/dev/benchmark/all-in-one/
```

## Configure YAML File

The contents of the yaml file are as follows:

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
- low_bit: Convert the model to which precision you want to test.
- batch_size: The number of samples on which the models makes predictions in one forward pass.
- in_out_pairs: Inputs given to the model and expected outputs. 
- test_api: Use different test functions on different machines.
  - `transformer_int4_gpu` on Intel GPU for Linux
  - `transformer_int4_gpu_win` on Intel GPU for Windows
  - `transformer_int4` on Intel CPU
- cpu_embedding: Whether put embedding to the CPU (only avaiable now for windows gpu related test_api)


## Run Script

Run the script, wait for the model to finish running, and then you can obtain a CSV result file whose name includes `test_api` and the date of the day under the current folder. You can check if the test results in CSV meet your expectations, such as `1st token avg latency (ms)` and `2+ avg latency (ms/token)`.

```
# install necessary dependencies
pip install tcmalloc
pip install pandas
pip install omegaconf
pip install wheel
pip install einops

# For ARC gpu, you can directly use `./run-arc.sh`.
# For MAX gpu, you can directly use `./run-max-gpu.sh`.
python run.py
```