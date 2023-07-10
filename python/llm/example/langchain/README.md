# Langchain examples

The examples here shows how to use langchain with `bigdl-llm`.

## Install bigdl-llm
Follow the instructions in  [bigdl-llm docs: Install]().

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U typing_extensions==4.5.0
```

Note that typing_extensions==4.5.0 is required, or you may encounter error `TypeError: dataclass_transform() got an unexpected keyword argument 'field_specifiers'` when running the examples. 


## Convert Models using bigdl-llm
Follow the instructions in [bigdl-llm docs: Convert Models]().


## Run the examples

### 1. Streaming Chat

```bash
python ./streamchat.py -m MODEL_PATH -x MODEL_FAMILY -t THREAD_NUM -q "What is AI?"
```
arguments info:
- `-m MODEL_PATH`: path to the converted model
- `-x MODEL_FAMILY`: the model family of the model specified in `-m`, available options are `llama`, `gptneox`
- `-q QUESTION `: question to ask. Default  is `What is AI?`.
- `-t THREAD_NUM`: required argument defining the number of threads to use for inference. Default is `2`.

### 2. Question Answering over Docs
```bash
python ./docqa.py --t THREAD_NUM -m -x
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: path to the converted model in above step
- `-x MODEL_FAMILY`: the model family of the model specified in `-m`, available options are `llama`, `gptneox`
- `-q QUESTION `: question to ask, default question is `What is AI?`.
- `-t THREAD_NUM`: required argument defining the number of threads to use for inference. Default is `2`.


