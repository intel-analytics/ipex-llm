# LLaVA

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on LLaVA models. For illustration purposes, we utilize the [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) as a reference LLaVA model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multi-turn chat centered around an image using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a LLaVA model to start a multi-turn chat centered around an image using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/haotian-liu/LLaVA.git # clone the llava libary
cd LLaVA # change the working directory to the LLaVA folder
git checkout tags/v1.2.0 -b 1.2.0 # Get the branch which is compatible with transformers 4.36
pip install -e . # Install llava
cd ..
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout tags/v1.2.0 -b 1.2.0
pip install -e .
cd ..
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the LLaVA model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py --image-path-or-url 'https://llava-vl.github.io/static/images/monalisa.jpg'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

If you encounter some network error (which means your machine is unable to access huggingface.co) when running this example, refer to [Trouble Shooting](#3-trouble-shooting) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --image-path-or-url 'https://llava-vl.github.io/static/images/monalisa.jpg'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the LLaVA model (e.g. `liuhaotian/llava-v1.5-13b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'liuhaotian/llava-v1.5-13b'`.
- `--image-path-or-url IMAGE_PATH_OR_URL`: argument defining the input image that the chat will focus on. It is required.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.


#### 2.4 Sample Output
#### [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)

```log
USER: 你知道这幅画是谁画的吗？
ASSISTANT: 这幅画是由著名的文艺复兴画家达芬奇（Leonardo da Vinci）画的。该画是他的代表作之一，是出自意大利佛罗伦萨的博物馆。画中的女子被认为是一位不为人知的模特，而且画作可能还有一个人物底版，这可能使得这幅画的价值更高。
```

The sample input image is:

<a href="https://llava-vl.github.io/static/images/monalisa.jpg"><img width=400px src="https://llava-vl.github.io/static/images/monalisa.jpg" ></a>

### 3. Trouble shooting

#### 3.1 SSLError
If you encounter the following output, it means your machine has some trouble accessing huggingface.co.
```log
requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /openai/clip-vit-large-patch14-336/resolve/main/config.json (Caused by SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1129)')))"),
```

You can resolve this problem with the following steps:
1. Download https://huggingface.co/openai/clip-vit-large-patch14-336 on some machine that can access huggingface.co, and put it in huggingface's local cache (default to be `~/.cache/huggingface/hub`) on the machine that you are going to run this example.
2. Set the environment variable (`export TRANSFORMERS_OFFLINE=1`) before you run the example.