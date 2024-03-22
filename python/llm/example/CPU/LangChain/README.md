## Langchain Examples

This folder contains examples showcasing how to use `langchain` with `bigdl`. 

### Install BigDL

Ensure `bigdl-llm` is installed by following the [BigDL-LLM Installation Guide](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install). 

### Install Dependences Required by the Examples


```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
```


### Example: Chat

The chat example ([chat.py](./transformers_int4/chat.py)) shows how to use `LLMChain` to build a chat pipeline. 

To run the example, execute the following command in the current directory:

```bash
python transformers_int4/chat.py -m <path_to_model> [-q <your_question>]
```
> Note: if `-q` is not specified, it will use `What is AI` by default. 

### Example: RAG (Retrival Augmented Generation) 

The RAG example ([rag.py](./transformers_int4/rag.py)) shows how to load the input text into vector database,  and then use `load_qa_chain` to build a retrival pipeline.

To run the example, execute the following command in the current directory:

```bash
python transformers_int4/rag.py -m <path_to_model> [-q <your_question>] [-i <path_to_input_txt>]
```
> Note: If `-i` is not specified, it will use a short introduction to Big-DL as input by default. if `-q` is not specified, `What is BigDL?` will be used by default. 


### Example: Math

The math example ([math.py](./transformers_int4/llm_math.py)) shows how to build a chat pipeline specialized in solving math questions. For example, you can ask `What is 13 raised to the .3432 power?`

To run the exmaple, execute the following command in the current directory:

```bash
python transformers_int4/llm_math.py -m <path_to_model> [-q <your_question>]
```
> Note: if `-q` is not specified, it will use `What is 13 raised to the .3432 power?` by default. 


### Example: Voice Assistant

The voice assistant example ([voiceassistant.py](./transformers_int4/voiceassistant.py)) showcases how to use langchain to build a pipeline that takes in your speech as input in realtime, use an ASR model (e.g. [Whisper-Medium](https://huggingface.co/openai/whisper-medium)) to turn speech into text, and then feed the text into large language model to get response.  

To run the exmaple, execute the following command in the current directory:

```bash
python transformers_int4/voiceassistant.py -m <path_to_model> [-q <your_question>]
```
**Runtime Arguments Explained**:
- `-m MODEL_PATH`: **Required**, the path to the 
- `-r RECOGNITION_MODEL_PATH`: **Required**,  the path to the huggingface speech recognition model
- `-x MAX_NEW_TOKENS`: the max new tokens of model tokens input
- `-l LANGUAGE`: you can specify a language such as "english" or "chinese" 
- `-d True|False`: whether the model path specified in -m is saved low bit model.


### Example: Streaming Chat
```bash
python transformers_int4/streamchat_native.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -q QUESTION -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-q QUESTION`: question to ask. Default is `What is AI?`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.


### Example: Question Answering over Docs
```bash
python transformers_int4/docqa_native.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -i DOC_PATH -q QUESTION -c CONTEXT_SIZE -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model in above step
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox` and `bloom`
- `-i DOC_PATH`: **required**, path to the input document
- `-q QUESTION`: question to ask. Default is `What is AI?`.
- `-c CONTEXT_SIZE`: specify the maximum context size. Default is `2048`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.


### Example: Voice Assistant - Native
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
python transformers_int4/voiceassistant_native.py -x MODEL_FAMILY -m CONVERTED_MODEL_PATH -t THREAD_NUM -c CONTEXT_SIZE
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
