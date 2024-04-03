# Run Ollama on Intel GPU

### 1 Install Ollama integrated with IPEX-LLM

First ensure that IPEX-LLM is installed. Follow the instructions on the [IPEX-LLM Installation Quickstart for Windows with Intel GPU](install_windows_gpu.html). And activate your conda environment.

Run `pip install --pre --upgrade ipex-llm[cpp]`, then execute `init-ollama`, you can see a softlink of `ollama`under your current directory.

### 2 Verify Ollama Serve

To avoid potential proxy issues, run `export no_proxy=localhost,127.0.0.1`. Execute `export ZES_ENABLE_SYSMAN=1` and `source /opt/intel/oneapi/setvars.sh` to enable driver initialization and dependencies for system management.

Start the service using `./ollama serve`. It should display something like:

![image-20240403164414684](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240403164414684.png)

To expose the `ollama` service port and access it from another machine, use `OLLAMA_HOST=0.0.0.0 ./ollama serve`.

Open another terminal, use `./ollama pull <model_name>` to download a model locally.

![image-20240403165342436](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240403165342436.png)

Verify the setup with the following command:

```shell
curl http://localhost:11434/api/generate -d '
{ 
  "model": "<model_name>", 
  "prompt": "Why is the sky blue?", 
  "stream": false 
}'
```

Expected results:

![image-20240403170520057](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240403170520057.png)

### 3 Example: Ollama Run

You can use `./ollama run <model_name>` to automatically pull and load the model for a stream chat.

![image-20240403165927706](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240403165927706.png)

![image-20240403170234524](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240403170234524.png)