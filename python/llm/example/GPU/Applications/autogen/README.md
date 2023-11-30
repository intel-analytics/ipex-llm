## Running AutoGen agent chat with BigDL-llm on Intel GPU
### 1. Install packages

FastChat
```
cd FastChat
pip3 install -e .
```

AutoGen
```
cd AutoGen
pip3 install -e .
```

BigDL-LLM
```
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### 2. Modification of FastChat and AutoGen

Add BigDL LLM adapter FastChat to load model. Set openai api_key and api_base for using FastChat API

### 3. Download LLM

This example used the existing llama2-chat-7b-hf model from the server and you can download other models from huggingface

### 4. Set up FastChat RESTful API for local LLM via BigDL

<h4>Steps:</h4>
- Open four terminals. Three for setting up API and one for testing
  - launch the controller, terminal 1 run
      ```
      cd FastChat
      python -m fastchat.serve.controller
      ```
      > Sample output: `Uvicorn running on http://localhost:21001 (Press CTRL+C to quit)`

  - launch the model worker(s), terminal 2 run 
      ```
      cd FastChat
      export no_proxy="localhost"
      python -m fastchat.serve.model_worker --model-path YOUR_LLM_PATH --device xpu --bigdl_load
      ```
      > Sample output: `127.0.0.1:41170 - "POST /worker_get_status HTTP/1.1" 200 OK`
  - launch the RESTful API server, terminal 3 run 
      ```
      cd FastChat
      python -m fastchat.serve.openai_api_server --host localhost --port 8000
      ```
      > Sample output: `127.0.0.1:34020 - "POST /v1/chat/completions HTTP/1.1" 200 OK`

  - test the api, termial 4 run 
    ```
    export no_proxy="localhost"
    curl http://localhost:8000/v1/models
    ```
    > Sample output: `{"object":"list","data":[{"id":"Llama-2-7b-chat-hf","object":"model",...` or `{"object":"list","data":[]}`



<h4>Bugs, Possible Cause and Fix:</h4>

**Agent Chat:**
- openai.error.ServiceUnavailableError: The server is overloaded or not ready yet.
  - Cause: Did not set `no_proxy="localhost"` when testing api
  - Fix: Set `no_proxy="localhost"`

- assert r.status_code == 200 / AssertionError
  - Cause: Did not set `no_proxy="localhost"` when setting up the model worker
  - Fix: Set `no_proxy="localhost"`

- openai.error.Timeout: Request timed out: HTTPConnectionPool(host='localhost', port=8000): Read timed out. (read timeout=60)
  - Cause: The inference time is too long and exceeded default
  - Fix: Set a larger `"request_timeout": 600,` in `llm_config` used to create assistant agent

- **Allocation is out of device memory on current platform.","code":50001' (HTTP response code was 400)
  - Cause: The multi-round agent chat stores chat history in the memory and cause out of memory
  - Fix: Set a smaller `max_consecutive_auto_reply` in `user_proxy` or change a better device. **Restart the server after getting this error. In my case, after getting this error, any further testing on the original server will not work**

- Only  allowed now, your model Llama-2-7b-chat-hf","code":40301 (HTTP response code was 400)
  - Cause: No idea yet, but it usually happens when I run the agent chat just after setting up the api
  - Fix: Wait for some time and run the agent chat again

- models&bc=Failed+to+retrieve+requested+URL.&ip=10.239.158.51&er=ERR_CONNECT_FAIL
  - Cause: Did not set `no_proxy="localhost"` when testing with `curl http://localhost:8000/v1/models`
  - Fix: Set `no_proxy="localhost"` in the corresponding terminal

**Chat Completion:**
- openai.error.AuthenticationError: No API key provided. You can set your API key in code using 'openai.api_key = <API-KEY>'
  - Cause: Did not set openai.api_key = "EMPTY" and openai.api_base = "http://localhost:8000/v1" for `openai.Completion`. **Notice: This is the cause only for the chat completion using local LLM, not for normal usage of openai api**
  - Fix: Set openai.api_key = "EMPTY" and openai.api_base = "http://localhost:8000/v1" in site-packages/autogen/oai/completion.py line 219 and line 220
