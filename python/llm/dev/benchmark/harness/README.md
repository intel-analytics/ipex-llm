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
```bash
export IPEX_LLM_LAST_LM_HEAD=0

python run_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device cpu --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
### Evaluation on Intel GPU
```bash
export IPEX_LLM_LAST_LM_HEAD=0

python run_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device xpu --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
### Evaluation using multiple Intel GPU
```bash
export IPEX_LLM_LAST_LM_HEAD=0

python run_multi_llb.py --model ipex-llm --pretrained /path/to/model --precision nf3 sym_int4 nf4 --device xpu:0,2,3 --tasks hellaswag arc mmlu truthfulqa --batch 1 --no_cache
```
Taking example above, the script will fork 3 processes, each for one xpu, to execute the tasks.
## Results
We follow [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) to record our metrics, `acc_norm` for `hellaswag` and `arc_challenge`, `mc2` for `truthful_qa` and `acc` for `mmlu`. For `mmlu`, there are 57 subtasks which means users may need to average them manually to get final result.
## Summarize the results
```python
python make_table.py <input_dir>
```

## Known Issues
### 1.Detected model is a low-bit(sym int4) model, please use `load_low_bit` to load this model
Harness evaluation is meant for unquantified models and by passing the argument `--precision` can the model be converted to target precision. If you load the quantified models, you may encounter the following error:
```bash
********************************Usage Error********************************
Detected model is a low-bit(sym int4) model, Please use load_low_bit to load this model.
```
 However, you can replace the following code in [this line](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/dev/benchmark/harness/ipexllm.py#L52):
```python
AutoModelForCausalLM.from_pretrained = partial(AutoModelForCausalLM.from_pretrained,**self.bigdl_llm_kwargs)
```
to the following codes to load the low bit models.
```python
class ModifiedAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def load_low_bit(cls,*args,**kwargs):
        for k in ['load_in_low_bit', 'device_map', 'max_memory','load_in_4bit']:
        kwargs.pop(k)
    return super().load_low_bit(*args, **kwargs)

AutoModelForCausalLM.from_pretrained=partial(ModifiedAutoModelForCausalLM.load_low_bit, *self.bigdl_llm_kwargs)
```
### 2.Please pass the argument `trust_remote_code=True` to allow custom code to be run.
`lm-evaluation-harness` doesn't pass `trust_remote_code=true` argument to datasets. This may cause errors similar to the following one:
```
RuntimeError: Job config of task=winogrande, precision=sym_int4 failed.
Error Message: The repository for winogrande contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/winogrande.
please pass the argument trust_remote_code=True to allow custom code to be run.
```
Please refer to these:

- [trust_remote_code error in simple evaluate for hellaswag · Issue #2222 · EleutherAI/lm-evaluation-harness (github.com) ](https://github.com/EleutherAI/lm-evaluation-harness/issues/2222)

- [Setting trust_remote_code to True for HuggingFace datasets compatibility by veekaybee · Pull Request #1467 · EleutherAI/lm-evaluation-harness (github.com)](https://github.com/EleutherAI/lm-evaluation-harness/pull/1467#issuecomment-1964282427)

- [Security features from the Hugging Face datasets library · Issue #1135 · EleutherAI/lm-evaluation-harness (github.com)](https://github.com/EleutherAI/lm-evaluation-harness/issues/1135#issuecomment-1961928695)

You have to manually run `export HF_DATASETS_TRUST_REMOTE_CODE=1` to solve the problem.

### 3.Error: xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,RuntimeError: unsupported dtype, only fp32 and fp16 are supported.
This error is because `ipex-llm` currently only support models with `torch_dtype` of `fp16` or `fp32`.

You can add `--model_args dtype=float16` to your command to solve this problem.