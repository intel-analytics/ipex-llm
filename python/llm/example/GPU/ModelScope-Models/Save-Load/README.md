# Save/Load Low-Bit Models with IPEX-LLM Optimizations

In this directory, you will find example on how you could save/load ModelScope models with IPEX-LLM INT4 optimizations on ModelScope models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [baichuan-inc/Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary) as a reference ModelScope model.

## 0. Requirements
To run this example with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../README.md#system-support) for more information.

## Example: Save/Load Model in Low-Bit Optimization
In the example [generate.py](./generate.py), we show a basic use case of saving/loading model in low-bit optimizations to predict the next N tokens using `generate()` API. Also, saving and loading operations are platform-independent, so you could run it on different platforms.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install modelscope==1.11.0
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9 libuv
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install modelscope==1.11.0
```

### 2. Configures OneAPI environment variables
#### 2.1 Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```

#### 2.2 Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.


### 3. Run
#### 3.1 Configurations for Linux
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

#### 3.2 Configurations for Windows
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

### 4. Running examples

If you want to save the optimized low-bit model, run:
```
python ./generate.py --save-path path/to/save/model
```

If you want to load the optimized low-bit model, run:
```
python ./generate.py --load-path path/to/load/model
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the ModelScope repo id for the Baichuan model to be downloaded, or the path to the ModelScope checkpoint folder. It is default to be `'baichuan-inc/Baichuan2-7B-Chat'`.
- `--save-path`: argument defining the path to save the low-bit model. Then you can load the low-bit directly.
- `--load-path`: argument defining the path to load low-bit model.
- `--prompt PROMPT`: argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [baichuan-inc/Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary)
```log
Inference time: xxxx s
-------------------- Output --------------------
<human>What is AI? <bot>Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that would typically require human intelligence. These tasks include learning, reasoning, problem
```
