# MiniCPM-V-2
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on MiniCPM-V-2 models. For illustration purposes, we utilize the [openbmb/MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2) as a reference MiniCPM-V-2 model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `chat()` API
In the example [chat.py](./chat.py), we show a basic use case for a MiniCPM-V-2 model to predict the next N tokens using `chat()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install peft timm
```
On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install peft timm
```

### 2. Run

- chat without streaming mode:
  ```
  python ./chat.py --prompt 'What is in the image?'
  ```
- chat in streaming mode:
  ```
  python ./chat.py --prompt 'What is in the image?' --stream
  ```

> [!TIP]
> For chatting in streaming mode, it is recommended to set the environment variable `PYTHONUNBUFFERED=1`.


Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the MiniCPM-V-2 model (e.g. `openbmb/MiniCPM-V-2`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openbmb/MiniCPM-V-2'`.
- `--image-url-or-path IMAGE_URL_OR_PATH`: argument defining the image to be infered. It is default to be `'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is in the image?'`.
- `--stream`: flag to chat in streaming mode

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the MiniCPM model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
python ./chat.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./chat.py
```

#### 2.3 Sample Output
#### [openbmb/MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2)
```log
Inference time: xxxx s
-------------------- Input Image --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Input Prompt --------------------
What is in the image?
-------------------- Chat Output --------------------
The image features a young child holding a white teddy bear dressed in pink. The background includes some red flowers and what appears to be a stone wall.
```

```log
-------------------- Input Image --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Input Prompt --------------------
图片里有什么？
-------------------- Stream Chat Output --------------------
图片中有一个小女孩，她手里拿着一个穿着粉色裙子的白色小熊玩偶。背景中有红色花朵和石头结构，可能是一个花园或庭院。
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>
