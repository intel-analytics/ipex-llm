# LangChain Example

The examples in this folder shows how to use [LangChain](https://www.langchain.com/) with `ipex-llm` on Intel CPU.

> [!TIP]
> For more information, please refer to the upstream LangChain LLM documentation with IPEX-LLM [here](https://python.langchain.com/docs/integrations/llms/ipex_llm), and upstream LangChain embedding model documentation with IPEX-LLM [here](https://python.langchain.com/docs/integrations/text_embedding/ipex_llm/).

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## 1. Install

We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:
```cmd
onda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
```

## 2. Run examples with LangChain

### 2.1. Example: Streaming Chat

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community
```

In the current directory, run the example with command:

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.

### 2.2. Example: Retrival Augmented Generation (RAG)

The RAG example ([rag.py](./rag.py)) shows how to load the input text into vector database, and then use LangChain to build a retrival pipeline.

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community langchain-chroma sentence-transformers==3.0.1
```

In the current directory, run the example with command:

```bash
python rag.py -m <path_to_llm_model> -e <path_to_embedding_model> [-q QUESTION] [-i INPUT_PATH]
```
**Additional Parameters for Configuration:**
- `-m LLM_MODEL_PATH`: **required**, path to the model.
- `-e EMBEDDING_MODEL_PATH`: **required**, path to the embedding model.
- `-q QUESTION`: question to ask. Default is `What is IPEX-LLM?`.
- `-i INPUT_PATH`: path to the input doc.


### 2.3. Example: Low Bit

The low_bit example ([low_bit.py](./low_bit.py)) showcases how to use use LangChain with low_bit optimized model.
By `save_low_bit` we save the weights of low_bit model into the target folder.
> [!NOTE]
> `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community
```

In the current directory, run the example with command:

```bash
python low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: question to ask. Default is `What is AI?`.

### 2.4. Example: Math

The math example ([math.py](./llm_math.py)) shows how to build a chat pipeline specialized in solving math questions. For example, you can ask `What is 13 raised to the .3432 power?`

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community
```

In the current directory, run the example with command:

```bash
python llm_math.py -m <path_to_model> [-q <your_question>]
```

**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the model
- `-q QUESTION`: question to ask. Default is `What is 13 raised to the .3432 power?`.

> [!NOTE]
> If `-q` is not specified, it will use `What is 13 raised to the .3432 power?` by default. 

### 2.5. Example: Voice Assistant

The voice assistant example ([voiceassistant.py](./voiceassistant.py)) showcases how to use LangChain to build a pipeline that takes in your speech as input in realtime, use an ASR model (e.g. [Whisper-Medium](https://huggingface.co/openai/whisper-medium)) to turn speech into text, and then feed the text into large language model to get response.  

Install LangChain dependencies:
```bash
pip install -U langchain langchain-community
pip install transformers==4.36.2
```

To run the exmaple, execute the following command in the current directory:

```bash
python voiceassistant.py -m <path_to_model> -r <path_to_recognition_model> [-q <your_question>]
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the 
- `-r RECOGNITION_MODEL_PATH`: **Required**,  the path to the huggingface speech recognition model
- `-x MAX_NEW_TOKENS`: the max new tokens of model tokens input
- `-l LANGUAGE`: you can specify a language such as "english" or "chinese" 
- `-d True|False`: whether the model path specified in -m is saved low bit model.
