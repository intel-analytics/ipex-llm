# Run Open-Webui with Ollama on Linux with Intel GPU

The [open-webui](https://github.com/open-webui/open-webui) provides a user friendly GUI for anyone to run LLM locally; by porting it to [ollama](https://github.com/ollama/ollama) integrated with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily run LLM in [open-webui](https://github.com/open-webui/open-webui) on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running Mistral:7B on Intel Arc A770 below.

<video src="" width="100%" controls></video>

## Quickstart
This quickstart guide walks you through setting up and using the [open-webui](https://github.com/open-webui/open-webui) with Ollama. 


### 1 Run Ollama on Linux with Intel GPU

To use the open-webui on Intel GPU, first ensure that you can run Ollama on Intel GPU. Follow the instructions on the [Run Ollama on Linux with Intel GPU](ollama_quickstart.md). Please keep ollama service running after completing the above steps.

### 2 Install and Run Open-Webui

#### Option 1: Docker

> [!IMPORTANT]
> Include `-v open-webui:/app/backend/data` in your Docker command when using Docker for Open WebUI. This ensures the database is properly mounted and prevents data loss.

- **For Ollama on Your Computer**:

  ```bash
  export no_proxy=localhost,127.0.0.1
  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- **For Ollama on a Different Server**:

  Replace `OLLAMA_BASE_URL` with your server's URL:

  ```bash
  export no_proxy=localhost,127.0.0.1
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- Access open-webui at [http://localhost:3000](http://localhost:3000). 

#### Option 2: Manual Build

**Requirements** 

- Node.js (>= 20.10) or Bun (>= 1.0.21)
- Python (>= 3.11)

**Installation Steps**

```sh
# Optional: Use Hugging Face mirror for restricted areas
export HF_ENDPOINT=https://hf-mirror.com

export no_proxy=localhost,127.0.0.1

git clone https://github.com/open-webui/open-webui.git
cd open-webui/
cp -RPp .env.example .env  # Copy required .env file

# Build Frontend
npm i
npm run build

# Serve Frontend with Backend
cd ./backend
pip install -r requirements.txt -U
bash start.sh
```

open-webui will be accessible at http://localhost:8080/.

> For detailed information, visit the [open-webui official repository](https://github.com/open-webui/open-webui).
>

### 3. Using Open-Webui

#### Log-in and Pull Model

If this is your first time using it, you need to register. After registering, log in with the registered account to access the interface.

![image-20240407170336578](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407170336578.png)

![image-20240407170703847](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407170703847.png)

Check your ollama service connection in `Settings`. If everything goes well, you will get a message as shown below.

![image-20240407170811631](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407170811631.png)

Pull model in `Settings/Models`, click the download button and ollama will download the model you select automatically.

![image-20240407171421875](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407171421875.png)

#### Chat with the Model

Start new conversations with **New chat**. Select a downloaded model here:

![image-20240407171643754](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407171643754.png)

Enter prompts into the textbox at the bottom and press the send button to receive responses.

![image-20240407172714188](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407172714188.png)

You can also drop files into the textbox for LLM to read.

![image-20240407172738371](C:\Users\yibopeng\AppData\Roaming\Typora\typora-user-images\image-20240407172738371.png)

#### Exit Open-Webui

To shut down the open-webui server, use **Ctrl+C** in the terminal where the open-webui server is runing, then close your browser tab.