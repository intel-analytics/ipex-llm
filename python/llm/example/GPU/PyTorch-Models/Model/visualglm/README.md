# VisualGLM

In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate VisualGLM-6b models. For illustration purposes, we utilize the [THUDM/visualglm-6b](https://huggingface.co/THUDM/visualglm-6b) as the reference VisualGLM models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Multi-turn chat centered around an image using chat() API
In the example [chat.py](./chat.py), we show a basic use case for a VisualGLM model to start a multi-turn chat centered around an image using `chat()`API, with BigDL-LLM INT4 optimizations on Intel GPUs.

### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

pip install SwissArmyTransformer # additional package required for VisualGLM to conduct generation
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install oneccl_bind_pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9 libuv
conda activate llm

pip install SwissArmyTransformer # additional package required for VisualGLM to conduct generation
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install oneccl_bind_pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu
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
### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
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

```bash
python ./chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --image-path IMAGE_PATH --n-predict N_PREDICT
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the VisualGLM model (e.g. `THUDM/visualglm-6b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/visualglm-6b'`.
- `--image-path`: argument defining the input image that the chat will focus on. It is required and should be a local path(not url).
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.

#### 4.1 Sample Output

The sample input image can be fetched from [COCO Dataset](https://cocodataset.org/#home)

[demo.jpg](https://cocodataset.org/#explore?id=70087)

<img src="http://farm8.staticflickr.com/7420/8726937863_e3bfa34795_z.jpg" width=50%>

```
用户: 介绍一下这幅图片
VisualGLM: 这张照片显示一辆白色公交车沿着一条街道行驶，穿过十字路口。在图像中可以看到几个人站在公交车站附近。公交车似乎正在等待乘客上车或下车。人们也可能正在等车经过他们所在的城市或目的地。总的来说，这幅画像描绘了一个交通繁忙的场景，包括公共交通和行人。
```
