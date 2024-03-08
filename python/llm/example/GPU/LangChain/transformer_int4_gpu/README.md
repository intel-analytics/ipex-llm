# Langchain examples

The examples in this folder shows how to use [LangChain](https://www.langchain.com/) with `bigdl-llm` on Intel GPU.

### 1. Install bigdl-llm
Follow the instructions in [GPU Install Guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) to install bigdl-llm

### 2. Install Required Dependencies for langchain examples. 

```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
```

### 3. Configures OneAPI environment variables
#### 3.1 Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```
#### 3.2 Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.
### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 4.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

#### 4.2 Configurations for Windows
<details>

<summary>For Intel iGPU</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

</details>

<details>

<summary>For Intel Arc™ A300-Series or Pro A60</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For other Intel dGPU Series</summary>

There is no need to set further environment variables.

</details>

> Note: For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

### 5. Run the examples

#### 5.1. Streaming Chat

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.


#### 5.2 Low Bit

### Example: Low Bit

The low_bit example ([low_bit.py](./transformers_int4/low_bit.py)) showcases how to use use langchain with low_bit optimized model.
By `save_low_bit` we save the weights of low_bit model into the target folder.
> Note: `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 

```bash
python transformers_int4/low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Runtime Arguments Explained**:
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: the question
