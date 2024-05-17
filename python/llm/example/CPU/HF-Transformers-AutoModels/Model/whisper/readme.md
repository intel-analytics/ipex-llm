# Whisper

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Whisper models. For illustration purposes, we utilize the [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) as a reference Whisper model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example 1: Recognize Tokens using `generate()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Whisper model to conduct transcription using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install datasets soundfile librosa # required by audio processing
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install datasets soundfile librosa
```

### 2. Run
```
python ./recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openai/whisper-tiny'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset to be downloaded, or the path to the huggingface dataset folder. It is default to be `'hf-internal-testing/librispeech_asr_dummy'`.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Whisper model based on the capabilities of your machine.


#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
python ./recognize.py 
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
#### [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)

```log
Inference time: xxxx s
-------------------- Output --------------------
[" Mr. Quilter is the Apostle of the Middle classes and we're glad to welcome his Gospel."]
```


## Example 2: Recognize Long Segment using `generate()` API
In the example [long-segment-recognize.py](./long-segment-recognize.py), we show a basic use case for a Whisper model to conduct transcription using `pipeline()` API for long audio input, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install datasets soundfile librosa # required by audio processing
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install datasets soundfile librosa # required by audio processing
```

### 2. Run
The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. For audio recordings longer than 30 seconds, it is possible to enable batched inference with `pipeline` method:
```
python ./long-segment-recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --audio-file PATH_TO_THE_AUDIO_FILE --language LANGUAGE --chunk-length CHUNK_LENGTH
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openai/whisper-medium'`.
- `--audio-file PATH_TO_THE_AUDIO_FILE`: argument defining the path of the audio file to be recognized.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.
- `--chunk-length CHUNK_LENGTH`: argument defining the maximum number of chuncks of sampling_rate samples used to trim and pad longer or shorter audio sequences. It is default to be 30, and chunk-length should not be larger than 30s for whisper model.
- `--batch-size`: argument defining the batch_size of pipeline inference, it usually equals of length of the audio divided by chunk-length. It is default to be 2.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Whisper model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
# Long Segment Recognize
python ./long-segment-recognize.py --audio-file /PATH/TO/AUDIO_FILE
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. long segment recognize for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./long-segment-recognize.py --audio-file /PATH/TO/AUDIO_FILE
```

#### 2.3 Sample Output
#### [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)

For audio file(.wav) download from https://www.youtube.com/watch?v=-LIIf7E-qFI, it should be extracted as:
```log
inference time is xxxx s
 I don't know who you are. I don't know what you want. If you're looking for ransom, I can tell you I don't have money. But what I do have are a very particular set of skills. Skills I have acquired over a very long career. Skills that make me a nightmare for people like you. If you let my daughter go now, that'll be the end of it. I will not look for you. I will not pursue you. But if you don't, I will look for you. I will find you. And I will kill you. Good luck.
```
