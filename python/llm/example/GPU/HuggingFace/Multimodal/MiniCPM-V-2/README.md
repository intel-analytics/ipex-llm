# MiniCPM-V-2
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on MiniCPM-V-2 model on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [openbmb/MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2) as a reference MiniCPM-V-2 model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `chat()` API
In the example [chat.py](./chat.py), we show a basic use case for a MiniCPM-V-2 model to predict the next N tokens using `chat()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install timm peft
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install timm peft
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
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

- chat without streaming mode:
  ```
  python ./chat.py --prompt 'What is in the image?'
  ```
- chat in streaming mode:
  ```
  python ./chat.py --prompt 'What is in the image?' --stream
  ```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the MiniCPM-V-2 (e.g. `openbmb/MiniCPM-V-2`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openbmb/MiniCPM-V-2'`.
- `--image-url-or-path IMAGE_URL_OR_PATH`: argument defining the image to be infered. It is default to be `'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is in the image?'`.
- `--stream`: flag to chat in streaming mode

#### Sample Output

#### [openbmb/MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2)

```log
Inference time: xxxx s
-------------------- Input Image --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Input Prompt --------------------
What is in the image?
-------------------- Chat Output --------------------
In the image, there is a young child holding a teddy bear. The teddy bear is dressed in a pink tutu. The child is also wearing a red and white striped dress. The background of the image features a stone wall and some red flowers.
```
```log
-------------------- Input Image --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Input Prompt --------------------
图片里有什么？
-------------------- Chat Output --------------------
图中是一个小女孩，她手里拿着一只粉白相间的泰迪熊。
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>
