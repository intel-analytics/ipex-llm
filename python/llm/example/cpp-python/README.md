# BigDL-LLM INT4 Inference Using Llama-Cpp-Python Format API

In this example, we show how to run inference on converted INT4 model using llama-cpp-python format API.

> **Note**: Currently model family LLaMA, GPT-NeoX, BLOOM and StarCoder are supported.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all]
```

## Convert Models using bigdl-llm
Follow the instructions in [bigdl-llm docs: Convert Models]().


## Run the example
```bash
python ./int4_inference.py -m CONVERTED_MODEL_PATH -x MODEL_FAMILY -p PROMPT -t THREAD_NUM
```
arguments info:
- `-m CONVERTED_MODEL_PATH`: **required**, path to the converted model
- `-x MODEL_FAMILY`: **required**, the model family of the model specified in `-m`, available options are `llama`, `gptneox`, `bloom` and `starcoder`
- `-p PROMPT`: question to ask. Default is `What is AI?`.
- `-t THREAD_NUM`: specify the number of threads to use for inference. Default is `2`.
