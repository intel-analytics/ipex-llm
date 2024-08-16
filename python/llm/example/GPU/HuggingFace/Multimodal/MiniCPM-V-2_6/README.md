# MiniCPM-V-2_6
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on MiniCPM-V-2_6 model on [Intel GPUs](../../../README.md). For illustration purposes, we utilize [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) as reference MiniCPM-V-2_6 model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `chat()` API
In the example [generate.py](./generate.py), we show a basic use case for a MiniCPM-V-2_6 model to predict the next N tokens using `chat()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install timm peft transformers==4.40.0 trl
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install timm peft transformers==4.40.0 trl
```

### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
export BIGDL_LLM_XMX_DISABLED=1
```

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

<summary>For Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

```
python ./generate.py --prompt 'What is in the image?'
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the MiniCPM-V-2_6 (e.g. `openbmb/MiniCPM-V-2_6`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openbmb/MiniCPM-V-2_6'`.
- `--image-url-or-path IMAGE_URL_OR_PATH`: argument defining the image to be infered. It is default to be `'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is in the image?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output

#### [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)

```log
Inference time: xxxx s
-------------------- Input --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Prompt --------------------
What is in the image?
-------------------- Output --------------------
The image features a young child holding a white teddy bear with a pink tutu. The child is wearing a striped dress and is standing in front of a stone wall with some red flowers in the background.
```
```log
Inference time: xxxx s
-------------------- Input --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Prompt --------------------
图片里有什么？
-------------------- Output --------------------
这幅图片展示了一个年幼的孩子，可能是一个蹒跚学步的幼儿，手里拿着一个毛绒玩具熊。孩子穿着一件条纹连衣裙，主要颜色是粉红色和白色。毛绒熊是白色的，戴着一条粉色的蝴蝶结围裙。背景中有红色的花朵，暗示着室外的环境，可能是一个花园或公园。
```
The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>
