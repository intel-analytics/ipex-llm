# Low-Bit Streaming LLM using IPEX-LLM

In this example, we apply low-bit optimizations to [Streaming-LLM](https://github.com/mit-han-lab/streaming-llm/tree/main#efficient-streaming-language-models-with-attention-sinks) using IPEX-LLM, which can deploy low-bit(including FP4/INT4/FP8/INT8) LLMs for infinite-length inputs.
Only one code change is needed to load the model using ipex-llm as follows:
```python
from ipex_llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True, trust_remote_code=True, optimize_model=False)
```

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
pip install -U transformers==4.34.0
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

## Run Example
```bash
python ./run_streaming_llama.py  --repo-id-or-model-path REPO_ID_OR_MODEL_PATH  --enable-streaming
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
