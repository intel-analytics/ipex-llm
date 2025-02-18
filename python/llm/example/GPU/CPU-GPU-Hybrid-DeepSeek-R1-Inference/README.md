# Run DeepSeek V3/R1 on Intel CPU and GPUs using IPEX-LLM

```bash
# From docker image: intelanalytics/ipex-llm-serving-xpu:2.2.0-b13
pip install transformers==4.48.3 accelerate trl==0.12.0

python generate.py --load-path /llm/models/DeepSeek-R1-int4-1/ --n-predict 128
```