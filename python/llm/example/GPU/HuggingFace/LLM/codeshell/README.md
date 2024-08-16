# CodeShell

In this directory, you'll find how to use this codeshell server with vscode codeshell extension.

## 0. Extra Environment Preparations

Suppose you have already configured GPU environment, you will need some extra preparation

1. install extra requirements
    ```
    pip install uvicorn fastapi sse_starlette
    ```

2. search `codeshell` in vscode extension market, then install `CodeShell VSCode Extension` extension

3. change extension settings:
    - change `Code Shell: Run Env For LLMs` to `GPU with TGI toolkit`
    - disable `Code Shell: Auto Trigger Completion` (use `Alt + \` to trigger completion manually)

4. download WisdomShell/CodeShell-7B-Chat (don't use CodeShell-7B)

## 1. How to use this server

This is a required step on Linux for APT. Skip this step for PIP-installed oneAPI or if you are running on Windows.
```bash
source /opt/intel/oneapi/setvars.sh
```

Then run the following command in the terminal:
```
python server.py [--option value]
```

1. `--checkpoint-path <path>`: path to huggingface model checkpoint
2. `--device xpu`: enable GPU or not
3. `--multi-turn`: enable multi turn conversation or just support single turn conversation
4. `--cpu-embedding`: move Embedding layer to CPU or not
5. `--max-context <number>`: Clip the context length in Code Completion, it won't affect other features, set it to 99999 to disable it

## 2. Note

In my test, if use vscode remote connection to connect to a remote machine, then install extension and running this server on that remote machine, all extension features expect for Code Completion can be used.

If don't use remote conection, then install extension and running this server on local machine, Code Completion can also be used.
