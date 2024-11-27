# Distil-Whisper

In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API on Distil-Whisper models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2) as a reference Distil-Whisper model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Recognize Tokens using `generate()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Distil-Whisper model to conduct transcription using `pipeline()` API for long audio input, with IPEX-LLM INT4 optimizations.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install datasets soundfile librosa # required by audio processing
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install datasets soundfile librosa # required by audio processing
```

### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

```
python ./recognize.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --repo-id-or-data-path REPO_ID_OR_DATA_PATH --language LANGUAGE --chunk-length CHUNK_LENGTH --batch-size BATCH_SIZE
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Distil-Whisper model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'distil-whisper/distil-large-v2'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset to be downloaded, or the path to the huggingface dataset folder. It is default to be `'distil-whisper/librispeech_long'`.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.
- `--chunk-length CHUNK_LENGTH`: argument defining the maximum number of chuncks of sampling_rate samples used to trim and pad longer or shorter audio sequences. For audio recordings less than 30 seconds, it can be set to 0 for better performance. It is default to be 15.
- `--batch-size BATCH_SIZE`: argument defining the batch_size of pipeline inference, it usually equals of length of the audio divided by chunk-length. It is default to be 16.

#### Sample Output
##### Short-Form Transcription

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

##### Long-Form Transcription

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