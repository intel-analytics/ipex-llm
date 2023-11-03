# VisualGLM

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on VisualGLM models. For illustration purposes, we utilize the [THUDM/visualglm-6b](https://huggingface.co/THUDM/visualglm-6b) as the reference VisualGLM models.

## Requirements

To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multi-turn chat centered around an image using chat() API

In the example [chat.py](./chat.py), we show a basic use case for a VisualGLM model to start a multi-turn chat centered around an image using `chat()`API, with BigDL-LLM INT4 optimizations.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latestbuild bigdl-llm nightly  with 'all' option
pip install SwissArmyTransformer torchvision cpm_kernels # additional package required for VisualGLM to conduct generation
git clone https://github.com/THUDM/SwissArmyTransformer
cd SwissArmyTransformer
pip install .
```

### 2. Run

```
python ./chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --image-path IMAGE_PATH --n-predict N_PREDICT
```

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the VisualGLM model (e.g. `THUDM/visualglm-6b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/visualglm-6b'`.
- `--image-path`: argument defining the input image that the chat will focus on. It is required and should be a local path(not url).
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.

#### 2.1 Client

On client Windows machine, it is recommended to run directly with full utilization of all cores:

```powershell
python ./chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --image-path IMAGE_PATH --n-predict N_PREDICT
```

#### 2.2 Server

For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,

```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output

The sample input image can be fetched from [COCO Dataset](https://cocodataset.org/#home)

[demo.jpg](https://cocodataset.org/#explore?id=70087)
<img src="http://farm8.staticflickr.com/7420/8726937863_e3bfa34795_z.jpg" width=50%>

```
用户: 介绍一下这幅图片
VisualGLM: 这张照片显示一辆白色公交车沿着一条街道行驶，穿过十字路口。在图像中可以看到几个人站在公交车站附近。公交车似乎正在等待乘客上车或下车。人们也可能正在等车经过他们所在的城市或目的地。总的来说，这幅画像描绘了一个交通繁忙的场景，包括公共交通和行人。
```
