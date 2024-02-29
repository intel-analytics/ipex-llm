# Langchain examples

The examples here show how to use langchain with `bigdl-llm`.

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
```

## Run the examples

### 1. Streaming Chat

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.
