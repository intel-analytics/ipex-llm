# LLaVA
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on LLaVA models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) as a reference LLaVA model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Multi-turn chat centered around an image using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a LLaVA model to start a multi-turn chat centered around an image using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

git clone -b v1.1.1 --depth=1 https://github.com/haotian-liu/LLaVA.git # clone the llava libary
pip install einops # install dependencies required by llava
cp generate.py ./LLaVA/ # copy our example to the LLaVA folder
cd LLaVA # change the working directory to the LLaVA folder
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```bash
python ./generate.py --image-path-or-url 'https://llava-vl.github.io/static/images/monalisa.jpg'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the LLaVA model (e.g. `liuhaotian/llava-v1.5-7b` to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'liuhaotian/llava-v1.5-7b'`.
- `--image-path-or-url IMAGE_PATH_OR_URL`: argument defining the input image that the chat will focus on. It is required.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.

If you encounter some network error (which means your machine is unable to access huggingface.co) when running this example, refer to [Trouble Shooting](#4-trouble-shooting) section.


#### Sample Output
#### [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)

```log
USER: Do you know who drew this painting?
ASSISTANT: Yes, the painting is a portrait of a woman by Leonardo da Vinci. It's a famous artwork known as the "Mona Lisa."
USER: Can you describe this painting?
ASSISTANT: The painting features a well-detailed portrait of a woman, painted in oil on a canvas. The woman appears to be a young woman staring straight ahead in a direct gaze towards the viewer. The woman's facial features are rendered sharply in the brush strokes, giving her a lifelike, yet enigmatic expression.
The background of the image mainly showcases the woman's face, with some hills visible in the lower part of the painting. The artist employs a wide range of shades, evoking a sense of depth and realism in the subject matter. The technique used in this portrait sets it apart from other artworks during the Renaissance period, making it a notable piece in art history.
```

The sample input image is:

<a href="https://llava-vl.github.io/static/images/monalisa.jpg"><img width=400px src="https://llava-vl.github.io/static/images/monalisa.jpg" ></a>

### 4 Trouble shooting

#### 4.1 SSLError
If you encounter the following output, it means your machine has some trouble accessing huggingface.co.
```log
requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /openai/clip-vit-large-patch14-336/resolve/main/config.json (Caused by SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1129)')))"),
```

You can resolve this problem with the following steps:
1. Download https://huggingface.co/openai/clip-vit-large-patch14-336 on some machine that can access huggingface.co, and put it in huggingface's local cache (default to be `~/.cache/huggingface/hub`) on the machine that you are going to run this example.
2. Set the environment variable (`export TRANSFORMERS_OFFLINE=1`) before you run the example.
