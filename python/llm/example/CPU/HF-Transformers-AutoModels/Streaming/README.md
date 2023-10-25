# BigDL-LLM Transformers Low-Bit Efficient Streaming Large Language Models with Attention Sinks

There are two major challenges to deploy Large Language Models (LLMs) in streaming applications where long interactions are expected. 
Firstly, there is a substantial memory overhead associated with caching the Key and Value states (KV) of previous tokens during the decoding process, resulting in application crashing from out of memory. 
Secondly, widely used LLMs can not generate longer text inputs than the training sequence length. [Xiao, 2023](https://arxiv.org/abs/2309.17453) discovered attention sink, 
that keeping the KV of initial tokens will largely recover the performance of window attention, and introduced (StreamingLLM)[https://github.com/mit-han-lab/streaming-llm/tree/main#efficient-streaming-language-models-with-attention-sinks], an efficient framework that enables LLMs to generalize to infinite sequence length.
In this example, we show the example of applying [efficient streaming with attention sinks](https://github.com/mit-han-lab/streaming-llm/tree/main#efficient-streaming-language-models-with-attention-sinks) on BigDL-LLM low-bit(including INT8/INT5/INT4) large language models. 

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all]
```

## Run Example
```bash
python ./run_streaming_llama.py  --repo-id-or-model-path REPO_ID_OR_MODEL_PATH  --enable_streaming
```
arguments info:
- `--repo-id-or-model-path`: str value, argument defining the huggingface repo id for the large language model to be downloaded, or the path to the huggingface checkpoint folder, the value is 'meta-llama/Llama-2-7b-chat-hf' by default.
- `--data-root`: str value, the directory to save downloaded questions data.
- `--enable-streaming`: to enable efficient streaming while computing.
- `--start-size`: int value, the start size of recent KV cache.
- `--recent-size`: optional str value. The path to load low-bit model.


## Sample Output for Inference
### 'decapoda-research/llama-7b-hf' Model
```log
USER: Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.

ASSISTANT: Dear Mr. Smith,

I am writing to seek your feedback on the 'Quarterly Financial Report' I prepared for the company. I have attached the report for your reference.
The report contains data analysis of the company's performance during the quarter ending 31st March 2019...
```
