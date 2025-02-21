# MiniCPM-o-2_6
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on MiniCPM-o-2_6 model on [Intel GPUs](../../../README.md). For illustration purposes, we utilize [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) as reference MiniCPM-o-2_6 model.

In the following examples, we will guide you to apply IPEX-LLM optimizations on MiniCPM-o-2_6 model for text/audio/image/video inputs.

## 0. Requirements & Installation

To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 0.1 Install IPEX-LLM

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)** on Windows:
  ```cmd
  conda create -n llm python=3.11 libuv
  conda activate llm

  :: or --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/cn/
  pip install --pre --upgrade ipex-llm[xpu_lnl] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/us/
  pip install torchaudio==2.3.1.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/us/
  ``` 
- For **Intel Arc B-Series GPU (code name Battlemage)** on Linux:
  ```cmd
  conda create -n llm python=3.11
  conda activate llm

  # or --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  pip install torchaudio==2.3.1+cxx11.abi --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ``` 

> [!NOTE]
> We will update for installation on more Intel GPU platforms.

###  0.2 Install Required Pacakges for MiniCPM-o-2_6

```bash
conda activate llm

# refer to: https://huggingface.co/openbmb/MiniCPM-o-2_6#usage
pip install transformers==4.44.2 trl
pip install librosa==0.9.0
pip install soundfile==0.12.1
pip install moviepy
```

### 0.3 Runtime Configuration

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)** on Windows:
  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  ``` 
- For **Intel Arc B-Series GPU (code name Battlemage)** on Linux:
  ```cmd
  unset OCL_ICD_VENDOR
  export SYCL_CACHE_PERSISTENT=1
  ``` 

> [!NOTE]
> We will update for runtime configuration on more Intel GPU platforms.

## 1. Example: Chat in Omni Mode
In [omni.py](./omni.py), we show a use case for a MiniCPM-V-2_6 model to chat in omni mode with IPEX-LLM INT4 optimizations on Intel GPUs. In this example, the model will take a video as input, and conduct inference based on the images and audio of this video.

For example, the video input shows a clip of an athlete swimming, with background audio asking "What the athlete is doing?". Then the model in omni mode should inference based on the images of the video and the question in audio.

### 1.1 Running example

```bash
python omni.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --video-path VIDEO_PATH
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for MiniCPM-o-2_6 model (e.g. `openbmb/MiniCPM-o-2_6`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openbmb/MiniCPM-o-2_6'`.
- `--video-path VIDEO_PATH`: argument defining the video input.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> [!NOTE]
> In Omni mode, please make sure that the video input contains sound.

> [!TIP]
> You could just ignore the warning regarding `Some weights of the model checkpoint at xxx were not used when initializing MiniCPMO`.

## 2. Example: Chat with text/audio/image input
In [chat.py](./chat.py), we show a use case for a MiniCPM-V-2_6 model to chat based on text/audio/image, or a combination of two of them, with IPEX-LLM INT4 optimizations on Intel GPUs.

### 2.1 Running example

- Chat with text input
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT
  ```

- Chat with audio input
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --audio-path AUDIO_PATH
  ```

- Chat with image input
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --image-path IMAGE_PATH
  ```

- Chat with text + audio inputs
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --audio-path AUDIO_PATH
  ```

- Chat with text + image inputs
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --image-path IMAGE_PATH
  ```

- Chat with audio + image inputs
  ```bash
  python chat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --audio-path AUDIO_PATH --image-path IMAGE_PATH
  ```


Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for MiniCPM-o-2_6 model (e.g. `openbmb/MiniCPM-o-2_6`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openbmb/MiniCPM-o-2_6'`.
- `--prompt PROMPT`: argument defining the text input.
- `--audio-path AUDIO_PATH`: argument defining the audio input.
- `--image-path IMAGE_PATH`: argument defining the image input.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> [!TIP]
> You could just ignore the warning regarding `Some weights of the model checkpoint at xxx were not used when initializing MiniCPMO`.

### 2.2 Sample Outputs

#### [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a><br>
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg

And the sample audio is a person saying "What is in this image".

- Chat with text + image inputs
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  5602445367_3504763978_z.jpg
  -------------------- Input Audio Path --------------------
  None
  -------------------- Input Prompt --------------------
  What is in this image?
  -------------------- Chat Output --------------------
  The image features a young child holding and displaying her white teddy bear. She is wearing a pink dress, which complements the color of the stuffed toy she
  ```

- Chat with audio + image inputs:
  ```log
  Inference time: xxxx s
  -------------------- Input Image Path --------------------
  5602445367_3504763978_z.jpg
  -------------------- Input Audio Path --------------------
  test_audio.wav
  -------------------- Input Prompt --------------------
  None
  -------------------- Chat Output --------------------
  In this image, there is a young girl holding and displaying her stuffed teddy bear. She appears to be the main subject of the photo, with her toy
  ```
