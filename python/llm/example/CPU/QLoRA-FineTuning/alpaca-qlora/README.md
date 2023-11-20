# Alpaca QLoRA Finetuning (experimental support)

This example ports [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) to BigDL-LLM QLoRA on [Intel CPUs](../../README.md).

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[all]
pip install transformers==4.34.0
pip install fire datasets peft==0.5.0
pip install accelerate==0.23.0
```

### 2. Configures environment variables
```bash
source bigdl-llm-init -t
```

### 3. Finetuning LLaMA-2-7B on a node:

Example usage:

```
python ./alpaca_qlora_finetuning_cpu.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca"
```

**Note**: You could also specify `--base_model` to the local path of the huggingface model checkpoint folder and `--data_path` to the local path of the dataset JSON file.

#### Sample Output
```log
{'loss': 1.9231, 'learning_rate': 2.9999945367033285e-05, 'epoch': 0.0}                                                                                                                            
{'loss': 1.8622, 'learning_rate': 2.9999781468531096e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.9043, 'learning_rate': 2.9999508305687345e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8967, 'learning_rate': 2.999912588049185e-05, 'epoch': 0.01}                                                                                                                            
{'loss': 1.9658, 'learning_rate': 2.9998634195730358e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8386, 'learning_rate': 2.9998033254984483e-05, 'epoch': 0.02}                                                                                                                           
{'loss': 1.809, 'learning_rate': 2.999732306263172e-05, 'epoch': 0.02}                                                                                                                             
{'loss': 1.8552, 'learning_rate': 2.9996503623845395e-05, 'epoch': 0.02}                                                                                                                           
  1%|█                                                                                                                                                         | 8/1164 [xx:xx<xx:xx:xx, xx s/it]
```

### Guide to use different prompts or different datasets
Now the prompter is for the datasets with `instruction` `input`(optional) and `output`. If you want to use different datasets,
you can add template file xxx.json in templates. And then update utils.prompter.py's `generate_prompt` method and update `generate_and_tokenize_prompt` method to fix the dataset.
For example, I want to train llama2-7b with [english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) just like [this example](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/CPU/QLoRA-FineTuning/qlora_finetuning_cpu.py)
1. add english_quotes.json
```json
{
    "prompt_no_input": "{quote} ->: {tags}"
}
```
2. update prompter.py
```python
def generate_prompt(self, quote: str, labels: Union[None, list]=None,) -> str:
    labels = str(labels)
    if input:
        res = self.template["prompt"].format(
            quote=quote, labels=labels
        )
    if self._verbose:
        print(res)
    return res
```
3. update generate_and_tokenize_prompt method
```python
def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["quote"], data_point["labels"]
    )
    user_prompt = prompter.generate_prompt(
        data_point["quote"], data_point["labels"]
    )
```