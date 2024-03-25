# IPEX-LLM Portable Zip with Web-UI For Windows: User Guide

## Introduction

This portable zip includes everything you need to run an LLM with IPEX-LLM optimizations and chat with it in Web-UI. Please refer to [How to use](#how-to-use) section to get started.

### 6B model running on an Intel 11-Gen Core PC (real-time screen capture)


### Verified Models

- ChatGLM2-6b

## How to use

1. Download the zip from link [here]().
2. (Optional) You could also build the zip on your own. Run `setup.bat --ui` and it will generate the zip file.
3. Unzip `ipex-llm.zip`.
4. Download the model to your computer.
5. Go into the unzipped folder and double click `chat-ui.bat`. Input the path of the model (e.g. `path\to\model`, note that there's no slash at the end of the path). Press Enter and wait until it shows `All service started. Visit 127.0.0.1:7860 in browser to chat.`. Do NOT close the terminal window!
6. Visit `127.0.0.1:7860` in your browser and enjoy chatting!
7. If you want to stop the program, just close the terminal window.