# IPEX-LLM Portable Zip For Windows: User Guide

## Introduction

This portable zip includes everything you need to run an LLM with IPEX-LLM optimizations (except models) . Please refer to [How to use](#how-to-use) section to get started.

### 13B model running on an Intel 11-Gen Core PC (real-time screen capture)

<p align="center">
   <a href="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-screen-capture.gif"><img src="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-screen-capture.gif" ></a>
</p>

### Verified Models
- Llama-2-7b-chat-hf
- Yi-6B-Chat
- Mixtral-8x7B-Instruct-v0.1
- Mistral-7B-Instruct-v0
- ChatGLM2-6b
- ChatGLM3-6b
- Baichuan-13B-Chat
- Baichuan2-7B-Chat
- internlm-chat-7b
- internlm2-chat-7b
- Qwen-7B-Chat

## How to use

1. Download the zip from link [here]().
2. (Optional) You could also build the zip on your own. Run `setup.bat` and it will generate the zip file.
3. Unzip `ipex-llm.zip`.
4. Download the model to your computer. Please ensure there is a file named `config.json` in the model folder, otherwise the script won't work.

<p align="center">
   <a href="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step1.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step1.png" ></a>
</p>

5. Go into the unzipped folder and double click `chat.bat`. Input the path of the model (e.g. `path\to\model`, note that there's no slash at the end of the path). Press Enter and wait until model finishes loading. Then enjoy chatting with the model!

<p align="center">
   <a href="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step2.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step2.png" ></a>
</p>

6. If you want to stop chatting, just input `stop` and the model will stop running.

<p align="center">
   <a href="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step34.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/one-click-installer-user-guide-step34.png" ></a>
</p>