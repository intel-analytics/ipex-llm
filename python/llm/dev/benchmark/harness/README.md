# Harness Evaluation
[Harness evaluation](https://github.com/EleutherAI/lm-evaluation-harness) allows users to eaisly get accuracy on various datasets. Here we have enabled harness evaluation with IPEX-LLM under 
[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) settings.
Before running, make sure to have [ipex-llm](../../../README.md) installed.

## Install Harness
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout b281b09
pip install -e .
```

## Run
run `python run_llb.py`. `run_llb.py` combines some arguments in `main.py` to make evaluations easier. The mapping of arguments is defined as a dict in [`llb.py`](llb.py).

### Evaluation on CPU
```python
python run_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device cpu --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
### Evaluation on Intel GPU
```python
python run_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device xpu --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
### Evaluation using multiple Intel GPU
```python
python run_multi_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device xpu:0,2,3 --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
Taking example above, the script will fork 3 processes, each for one xpu, to execute the tasks.
## Results
We follow [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) to record our metrics, `acc_norm` for `hellaswag` and `arc_challenge`, `mc2` for `truthful_qa` and `acc` for `mmlu`. For `mmlu`, there are 57 subtasks which means users may need to average them manually to get final result.
## Summarize the results
```python
python make_table.py <input_dir>
```
