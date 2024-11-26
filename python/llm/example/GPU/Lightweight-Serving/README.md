# Running Lightweight Serving using IPEX-LLM on one Intel GPU

## Requirements

To run this example with IPEX-LLM on one Intel GPU, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example

### 1. Install

#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install fastapi uvicorn openai
pip install gradio # for gradio web UI
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc

# for internlm-xcomposer2-vl-7b
pip install transformers==4.31.0
pip install accelerate timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops

# for whisper-large-v3
pip install transformers==4.36.2
pip install datasets soundfile librosa # required by audio processing
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install fastapi uvicorn openai
pip install gradio # for gradio web UI
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc

# for glm-4v-9b
pip install transformers==4.42.4 "trl<0.12.0"

# for internlm-xcomposer2-vl-7b
pip install transformers==4.31.0
pip install accelerate timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops

# for whisper-large-v3
pip install transformers==4.36.2
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

### 4. Running example

```
python ./lightweight_serving.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --low-bit LOW_BIT --port PORT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--low-bit LOW_BIT`: Sets the low bit optimizations (such as 'sym_int4', 'fp16', 'fp8' and 'fp6') for the model. It is default to be `sym_int4`.
- `--port PORT`: The serving access port. It is default to be `8000`.


### 5. Sample Input and Output

We can use `curl` to test serving api. And need to set no_proxy to ensure that requests are not forwarded by a proxy. `export no_proxy=localhost,127.0.0.1`

#### /generate

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "inputs": "What is AI?",
  "parameters": {
    "max_new_tokens": 32,
    "min_new_tokens": 32,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "do_sample": false,
    "top_k": 5,
    "tok_p": 1.0
  },
  "stream": false
}' http://localhost:8000/generate
```

#### /generate_stream

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "inputs": "What is AI?",
  "parameters": {
    "max_new_tokens": 32,
    "min_new_tokens": 32,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "do_sample": false,
    "top_k": 5,
    "tok_p": 1.0
  },
  "stream": false
}' http://localhost:8000/generate_stream
```

#### /v1/chat/completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}],
    "stream": false
  }'
```

##### Image input

image input only supports [internlm-xcomposer2-vl-7b](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) and [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) now. And they should both install specific transformers version to run.
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm-xcomposer2-vl-7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What'\''s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
```

#### /v1/completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-2-7b-chat-hf",
    "prompt": "Once upon a time",
    "max_tokens": 32,
    "stream": false
  }'
```

#### v1/audio/transcriptions

ASR only supports [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) now. And `whisper-large-v3` just can be used to transcription audio. The audio file_type should be supported by `librosa.load`.
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@/llm/test.mp3" \
  -F model="whisper-large-v3" \
  -F languag="zh"
```

### 6. Benchmark with wrk

Please refer to [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Pipeline-Parallel-Serving#4-benchmark-with-wrk) for more details

## 7. Using the `benchmark.py` Script

Please refer to [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Pipeline-Parallel-Serving#5-using-the-benchmarkpy-script) for more details

## 8. Gradio Web UI

Please refer to [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Pipeline-Parallel-Serving#6-gradio-web-ui) for more details