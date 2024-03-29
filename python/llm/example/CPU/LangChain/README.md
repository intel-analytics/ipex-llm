## Langchain Examples

This folder contains examples showcasing how to use `langchain` with `ipex-llm`. 

### Install-IPEX LLM

Ensure `ipex-llm` is installed by following the [IPEX-LLM Installation Guide](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm#install). 

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
> Note: If `-i` is not specified, it will use a short introduction to Big-DL as input by default. if `-q` is not specified, `What is IPEX LLM?` will be used by default. 


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


### Example: Low Bit

The low_bit example ([low_bit.py](./low_bit.py)) showcases how to use use langchain with low_bit optimized model.
By `save_low_bit` we save the weights of low_bit model into the target folder.
> Note: `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 
```bash
python low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Runtime Arguments Explained**:
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: the question


### Legacy (Native INT4 examples)

IPEX-LLM also provides langchain integrations using native INT4 mode. Those examples can be foud in [native_int4](./native_int4/) folder. For detailed instructions of settting up and running `native_int4` examples, refer to [Native INT4 Examples README](./README_nativeint4.md). 

