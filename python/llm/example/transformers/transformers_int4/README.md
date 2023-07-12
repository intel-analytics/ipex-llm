# BigDL-LLM Transformers INT4 Optimization for Large Language Model
BigDL-LLM supports INT4 optimizations to any Hugging Face Transformers models.

In this directory, we provide several model-specific examples with BigDL-LLM INT4 optimizations. You can navigate to the folder corresponding to the model you want to use.

## Recommended Requirements
To run the examples, we recommend using Intel® Xeon® processors (server), or >= 12th Gen Intel® Core™ processor (client).

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all]
```

## Best Known Configuration
For better performance, it is recommended to set environment variables set by BigDL-Nano:
```bash
pip install bigdl-nano
```
following with
| Linux | Windows (powershell)|
|:------|:-------|
|`source bigdl-nano-init`|`bigdl-nano-init`|

To better utilize multiple cores for improved performance, consider the following rules:
1. On server, it is recommended to utilize all the physical cores of a single socket. E.g. on Linux,
   ```bash
   # for a server with 48 cores per socket
   export OMP_NUM_THREADS=48
   numactl -C 0-47 -m 0 python -u example.py
   ```
2. On client machine, it is recommended to use all the performnace-cores along with their hyperthreads. E.g. on Windows,
   ```powershell
   # for a client machine with 8 Performance-cores
   $env:OMP_NUM_THREADS=16
   ```