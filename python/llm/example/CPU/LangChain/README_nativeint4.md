# Langchain Native INT4 examples

The examples in [native_int4](./native_int4) folder show how to use langchain with `bigdl-llm` native INT4 mode.

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
```


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

Please say something through your microphone (e.g. What is AI). The program will automatically detect when you have completed your speech and recognize them.

#### Known Issues
The speech_recognition library may occasionally skip recording due to low volume. An alternative option is to save the recording in WAV format using `PyAudio` and read the file as an input. Here is an example using PyAudio:
```python
import pyaudio
import speech_recognition as sr

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1                # The desired number of input channels
RATE = 16000                # The desired rate (in Hz)
RECORD_SECONDS = 10         # Recording time (in second)
WAVE_OUTPUT_FILENAME = "/path/to/pyaudio_out.wav"
p = pyaudio.PyAudio()
                
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("*"*10, "Listening\n")
frames = []
data =0
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
  data = stream.read(CHUNK)  ## <class 'bytes'> ,exception_on_overflow = False
  frames.append(data)   ## <class 'list'>
print("*"*10, "Stop recording\n")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

r = sr.Recognizer()
with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source1:
    audio = r.record(source1)  # read the entire audio file   
frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
```

### 4. Math

This is an example using `LLMMathChain`. This example has been validated using [phoenix-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b).

```bash
python transformers_int4/math.py -m MODEL_PATH -q QUESTION
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the transformers model
- `-q QUESTION`: question to ask. Default is `What is 13 raised to the .3432 power?`.

