# Distil-Whisper

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Distil-Whisper models. For illustration purposes, we utilize the [distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2) as a reference Distil-Whisper model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Recognize Tokens using `generate()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Distil-Whisper model to conduct transcription using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

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
python ./recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE --chunk-length CHUNK_LENGTH --batch-size BATCH_SIZE
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Distil-Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'distil-whisper/distil-large-v2'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset to be downloaded, or the path to the huggingface dataset folder. It is default to be `'distil-whisper/librispeech_long'`.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.
- `--chunk-length CHUNK_LENGTH`: argument defining the maximum number of chuncks of sampling_rate samples used to trim and pad longer or shorter audio sequences. For audio recordings less than 30 seconds, it can be set to 0 for better performance. It is default to be 15.
- `--batch-size BATCH_SIZE`: argument defining the batch_size of pipeline inference, it usually equals of length of the audio divided by chunk-length. It is default to be 16.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Distil-Whisper model based on the capabilities of your machine.


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
#### 2.3.1 Short-Form Transcription

Model: [distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)

Command line:
```bash
python ./recognize.py --repo-id-or-data-path 'hf-internal-testing/librispeech_asr_dummy' --chunk-length 0
```
Output:
```log
Inference time: xxxx s
-------------------- Output --------------------
[' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.']
```

#### 2.3.2 Long-Form Transcription

Model: [distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)

Command line:
```bash
python ./recognize.py --repo-id-or-data-path 'distil-whisper/librispeech_long' --chunk-length 15
```
Output:
```log
inference time is xxxx s
 Mr Quilter is the Apostle of the Middle classes, and we are glad to welcome his Gospel. Nor is Mr Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Leighton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linel's pictures are a sort of upguards and Adam paintings, and Mason's exquisite itels are as national as a Jingo poem. Mr Birkett Foster's landscapes smile at one much in the same way that Mr. Karker used to flash his teeth, and Mr. John Collier gives his sitter a cheerful slap on the back before he says, like a shampoo or a Turkish bath, next man.
```
