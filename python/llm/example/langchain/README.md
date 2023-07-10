# Langchain examples

The examples here shows how to use langchain with `bigdl-llm`.

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U typing_extensions==4.5.0
```

Note that typing_extensions==4.5.0 is required, or you may encounter error `TypeError: dataclass_transform() got an unexpected keyword argument 'field_specifiers'` when running the examples. 


## Convert Models using bigdl-llm
Follow the instructions in [Convert model](https://github.com/intel-analytics/BigDL/tree/main/python/llm#convert-model).


## Run the examples

### 1. Streaming Chat

```bash
python ./streamchat.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -q QUESTION -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-q QUESTION`: question to ask. Default is `What is AI?`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.

### 2. Question Answering over Docs
```bash
python ./docqa.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -i DOC_PATH -q QUESTION -c CONTEXT_SIZE -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model in above step
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-i DOC_PATH`: **required**, path to the input document
- `-q QUESTION`: question to ask. Default is `What is AI?`.
- `-c CONTEXT_SIZE`: specify the maximum context size. Default is `2048`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.

### 3. Voice Assistant
> This example is adapted from https://python.langchain.com/docs/use_cases/chatbots/voice_assistant with only tiny code change.

Some extra dependencies are required to be installed for this example.
```bash
pip install SpeechRecognition
pip install pyttsx3
pip install PyAudio
pip install whisper.ai
pip install soundfile
```

```bash
python ./voiceassistant.py -x MODEL_FAMILY -m CONVERTED_MODEL_PATH -t THREAD_NUM
```

arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.

When you see output says
> listening now...

Please say something through your microphone (e.g. What is AI). The programe will automatically detect when you have completed your speech and recogize them.
