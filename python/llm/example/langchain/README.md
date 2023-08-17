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
python native_int4/streamchat.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -q QUESTION -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-q QUESTION`: question to ask. Default is `What is AI?`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.

### 2. Question Answering over Docs
```bash
python native_int4/docqa.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -i DOC_PATH -q QUESTION -c CONTEXT_SIZE -t THREAD_NUM
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
python native_int4/voiceassistant.py -x MODEL_FAMILY -m CONVERTED_MODEL_PATH -t THREAD_NUM -c CONTEXT_SIZE
```

arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.
- `-c CONTEXT_SIZE`: specify maximum context size. Default to be 512.

When you see output says
> listening now...

Please say something through your microphone (e.g. What is AI). The programe will automatically detect when you have completed your speech and recogize them.

### 4. Math

This is an example using `LLMMathChain`. This example has been validated using [phoenix-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b).

```bash
python transformers_int4/math.py -m MODEL_PATH -q QUESTION
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the transformers model
- `-q QUESTION`: question to ask. Default is `What is 13 raised to the .3432 power?`.

