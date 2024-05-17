# Whisper

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on general pytorch models, for example Openai Whisper models. For illustration purposes, we utilize the [whisper-tiny](https://github.com/openai/whisper/blob/main/model-card.md) as a reference Whisper model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Recognize Tokens using `transcribe()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Whisper model to conduct transcription using `transcribe()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U openai-whisper
pip install librosa # required by audio processing
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install -U openai-whisper
pip install librosa
```

### 2. Run
```
python ./recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE
```

Arguments info:
- `--model-name MODEL_NAME`: argument defining the model name(tiny, medium, base, etc.) for the Whisper model to be downloaded. It is one of the official model names listed by `whisper.available_models()`, or path to a model checkpoint containing the model dimensions and the model state_dict. It is default to be `'tiny'`.
- `--audio-file AUDIO_FILE`: argument defining the path of the audio file to be recognized.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Whisper model based on the capabilities of your machine.


#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
python ./recognize.py --audio-file /PATH/TO/AUDIO_FILE
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./recognize.py
```

#### 2.3 Sample Output
#### [whisper-tiny](https://github.com/openai/whisper/blob/main/model-card.md)

For audio file(.wav) download from https://www.youtube.com/watch?v=-LIIf7E-qFI, it should be extracted as:
```log
[00:00.000 --> 00:10.000]  I don't know who you are.
[00:10.000 --> 00:15.000]  I don't know what you want.
[00:15.000 --> 00:21.000]  If you're looking for ransom, I can tell you I don't know money, but what I do have.
[00:21.000 --> 00:24.000]  I'm a very particular set of skills.
[00:24.000 --> 00:27.000]  The skills I have acquired are very long career.
[00:27.000 --> 00:31.000]  The skills that make me a nightmare for people like you.
[00:31.000 --> 00:35.000]  If you let my daughter go now, that'll be the end of it.
[00:35.000 --> 00:39.000]  I will not look for you. I will not pursue you.
[00:39.000 --> 00:45.000]  But if you don't, I will look for you. I will find you.
[00:45.000 --> 00:48.000]  And I will kill you.
[00:48.000 --> 00:53.000]  Good luck.
Inference time: xxxx s
-------------------- Output --------------------
 I don't know who you are. I don't know what you want. If you're looking for ransom, I can tell you I don't know money, but what I do have. I'm a very particular set of skills. The skills I have acquired are very long career. The skills that make me a nightmare for people like you. If you let my daughter go now, that'll be the end of it. I will not look for you. I will not pursue you. But if you don't, I will look for you. I will find you. And I will kill you. Good luck.
```
