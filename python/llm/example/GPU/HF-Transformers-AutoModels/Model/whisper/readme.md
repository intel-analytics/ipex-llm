# Whisper

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Whisper models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) as a reference Whisper model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Recognize Tokens using `generate()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Whisper model to conduct transcription using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install datasets soundfile librosa # required by audio processing
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run
```
python ./recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openai/whisper-tiny'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset to be downloaded, or the path to the huggingface dataset folder. It is default to be `'hf-internal-testing/librispeech_asr_dummy'`.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.

#### Sample Output
#### [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)

```log
Inference time: xxxx s
-------------------- Output --------------------
[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
```
