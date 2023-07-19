# Whisper

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Whisper models. For illustration purposes, we utilize the [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) as a reference Whisper model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Whisper model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'openai/whisper-tiny'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset to be downloaded, or the path to the huggingface dataset folder. It is default to be `'hf-internal-testing/librispeech_asr_dummy'`.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Whisper model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
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
#### [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)

```log
Inference time: 0.23290777206420898 s
-------------------- Output --------------------
[" Mr. Quilter is the Apostle of the Middle classes and we're glad to welcome his Gospel."]
```