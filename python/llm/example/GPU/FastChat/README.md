# Using FastChat bigdl-worker to run serving on Intel iGPU

This example demonstrates how to serve a BaiChuan2-7B model using [FastChat](https://github.com/lm-sys/FastChat) bigdl-worker Intel iGPU (with BigDL-LLM low-bits optimizations).

### 1. Install bigdl-llm

Follow the instructions in [GPU Install Guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) to install bigdl-llm

### 2. Install FastChat using pip

```bash
git clone -b new_bigdl_worker https://github.com/analytics-zoo/FastChat.git # Our bigdl worker has not been merged into FastChat
conda activate llm # Activate the environment in which BigDL-LLM is installed.
cd FastChat
pip install -e ".[model_worker,webui]"
```

### 3. Configures OneAPI environment variables

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" # Configurations for Windows
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

### 4. Serving with web GUI

To serve using the web UI, you need three main components: web servers that interface with users, bigdl model workers that host one or more models, and a controller to coordinate the webserver and model workers.

#### 4.1 Launch the controller

```bash
python3 -m fastchat.serve.controller
```

#### 4.2 Launch the bigdl model worker(s)

We should replace the normal worker (`fastchat.serve.model_worker`) with the bigdl worker (`fastchat.serve.bigdl_worker`).

```bash
# Change model-path to your model storage location
# Available low_bit format including sym_int4, sym_int8, bf16 etc.
# For intel iGPU, set device to "xpu"
python3 -m fastchat.serve.bigdl_worker --model-path D:\llm-models\Baichuan2-7B-Chat --low-bit "sym_int4" --trust-remote-code --device "xpu" 
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

#### 4.3 Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

Wait until you see "Running on local URL ...". You can access the web UI at <http://localhost:7860>.

#### (Optional) 4.4 Using OpenAI-Compatible RESTful APIs

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

