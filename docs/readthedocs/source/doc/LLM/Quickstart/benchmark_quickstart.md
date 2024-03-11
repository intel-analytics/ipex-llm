# BigDL-LLM Benchmarking

We can do benchmarking for BigDL-LLM on Intel CPUs and GPUs using the benchmark scripts we provide.

## Prepare The Environment

You can refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install.html) to install BigDL-LLM in your environment. The following dependencies are also needed to run the benchmark scripts.

```
pip install pandas
pip install omegaconf
```

## Prepare The Scripts

Navigate to your local workspace and then download BigDL from GitHub. Modify the `config.yaml` under `all-in-one` folder for your own benchmark configurations.

```
cd your/local/workspace
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/python/llm/dev/benchmark/all-in-one/
```

## Configure YAML File

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
  - "transformer_int4_gpu"
cpu_embedding: False
```

Some parameters in the yaml file that you can configure:

- repo_id: The name of the model and its organization.
- local_model_hub: The folder path where the models are stored on your machine.
- low_bit: The low_bit precision you want to convert to for benchmarking.
- batch_size: The number of samples on which the models makes predictions in one forward pass.
- in_out_pairs: Input sequence length and output sequence length combined by '-'.
- test_api: Use different test functions on different machines.
  - `transformer_int4_gpu` on Intel GPU for Linux
  - `transformer_int4_gpu_win` on Intel GPU for Windows
  - `transformer_int4` on Intel CPU
- cpu_embedding: Whether to put embedding on CPU (only avaiable now for windows gpu related test_api).

## Run on Windows

Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration) to configure oneAPI environment variables.

```eval_rst
.. tabs::
   .. tab:: Intel iGPU

      .. code-block:: bash

         set SYCL_CACHE_PERSISTENT=1
         set BIGDL_LLM_XMX_DISABLED=1

         python run.py

   .. tab:: Intel Arc™ A300-Series or Pro A60

      .. code-block:: bash

         set SYCL_CACHE_PERSISTENT=1
         python run.py

   .. tab:: Other Intel dGPU Series

      .. code-block:: bash
         # e.g. Arc™ A770
         python run.py

```

## Run on Linux

```eval_rst
.. tabs::
   .. tab:: Intel Arc™ A-Series and Intel Data Center GPU Flex

      For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:

      .. code-block:: bash

         ./run-arc.sh

   .. tab:: Intel Data Center GPU Max

      For Intel Data Center GPU Max Series, we recommend:

      .. code-block:: bash

         ./run-max-gpu.sh

      Please note that you need to run ``conda install -c conda-forge -y gperftools=2.10`` to install essential dependencies for Intel Data Center GPU Max.

```

## Result

After the script runnning is completed, you can obtain a CSV result file under the current folder. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for  performance results. You can also check whether the column `actual input/output tokens` is consistent with the column `input/output tokens` and whether the parameters you specified in config.yaml have been successfully applied in the benchmarking.