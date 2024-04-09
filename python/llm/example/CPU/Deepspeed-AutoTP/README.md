### Run Tensor-Parallel IPEX-LLM Transformers INT4 Inference with Deepspeed

#### 1. Install Dependencies

Install necessary packages (here Python 3.11 is our test environment):

```bash
bash install.sh
```

The first step in the script is to install oneCCL (wrapper for Intel MPI) to enable distributed communication between deepspeed instances, which can be skipped if Inte MPI/oneCCL/oneAPI has already been prepared on your machine. Please refer to [oneCCL](https://github.com/oneapi-src/oneCCL) if any related issue when install or import.

#### 2. Initialize Deepspeed Distributed Context

Like shown in example code `deepspeed_autotp.py`, you can construct parallel model with Python API:

```python
# Load in HuggingFace Transformers' model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(...)


# Parallelize model on deepspeed
import deepspeed

model = deepspeed.init_inference(
    model, # an AutoModel of Transformers
    mp_size = world_size, # instance (process) count
    dtype=torch.float16,
    replace_method="auto")
```

Then, returned model is converted into a deepspeed InferenceEnginee type.

#### 3. Optimize Model with IPEX-LLM Low Bit

Distributed model managed by deepspeed can be further optimized with IPEX low-bit Python API, e.g. sym_int4:

```python
# Apply IPEX-LLM INT4 optimizations on transformers
from ipex_llm import optimize_model

model = optimize_model(model.module.to(f'cpu'), low_bit='sym_int4')
model = model.to(f'cpu:{local_rank}') # move partial model to local rank
```

Then, a ipex-llm transformers is returned, which in the following, can serve in parallel with native APIs.

#### 4. Start Python Code

You can try deepspeed with IPEX LLM by:

```bash
bash run.sh
```

If you want to run your own application, there are **necessary configurations in the script** which can also be ported to run your custom deepspeed application:

```bash
# run.sh
source ipex-llm-init
unset OMP_NUM_THREADS # deepspeed will set it for each instance automatically
source /opt/intel/oneccl/env/setvars.sh
......
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_PROCESS_LAUNCHER=none
```

Set the above configurations before running `deepspeed` please to ensure right parallel communication and high performance.
