# Harness Evalution
Harness evalution allows users to eaisly get accuracy on various datasets. Here we enabled harness evalution with BigDL-LLM under 
[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) settings.
Before running, make sure to have [bigdl-llm](../../../README.md) and [bigdl-nano](../../../../nano/README.md) installed.

## Install Harness
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd  lm-evaluation-harness
git checkout e81d3cc
pip install -e .
git apply ../bigdl-llm.patch
cd ..
```

## Run
run `python llb.py`. `llb.py` combines some arguments in `main.py` to make evalutions easier. The mapping of arguments is defined as a dict in [`llb.py`](llb.py).

### CPU Usage
```python
python llb.py --model bigdl-llm --pretrained /path/to/model --precision nf3 int4 nf4 --device cpu --tasks hellaswag arc mmlu truthfulqa --output_dir results/output
```
### Intel GPU Usage
```python
python llb.py --model bigdl-llm --pretrained /path/to/model --precision nf3 int4 nf4 --device xpu --tasks hellaswag arc mmlu truthfulqa --output_dir results/output
```
## Results
We follow [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) to record our metrics, `acc_norm` for `hellaswag` and `arc_challenge`, `mc2` for `truthful_qa` and `acc` for `mmlu`. For `mmlu`, there are 57 subtasks which means users may need to average them manually to get final result.
