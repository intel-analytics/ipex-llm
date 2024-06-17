# Alpaca QLoRA Finetuning (experimental support)

This example ports [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) to IPEX-LLM QLoRA on [Intel CPUs](../../README.md).

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all]
pip install datasets transformers==4.36.0
pip install fire peft==0.10.0
pip install bitsandbytes scipy
```

### 2. Configures environment variables

```bash
source ipex-llm-init -t
```

### 3. Finetuning LLaMA-2-7B on a node:

Example usage:

```
python ./alpaca_qlora_finetuning_cpu.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./ipex-llm-qlora-alpaca"
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

### Guide to finetuning QLoRA on one node with multiple sockets

1. install extra lib

```bash
# need to run the alpaca stand-alone version first
# for using mpirun
pip install oneccl_bind_pt --extra-index-url https://developer.intel.com/ipex-whl-stable
```

2. modify conf in `finetune_one_node_two_sockets.sh` and run

```
source ${conda_env}/lib/python3.11/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
bash finetune_one_node_two_sockets.sh
```

### Guide to use different prompts or different datasets

Now the prompter is for the datasets with `instruction` `input`(optional) and `output`. If you want to use different datasets,
you can add template file xxx.json in templates. And then update utils.prompter.py's `generate_prompt` method and update `generate_and_tokenize_prompt` method to fix the dataset.
For example, I want to train llama2-7b with [english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) just like [this example](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/QLoRA-FineTuning/qlora_finetuning_cpu.py)

1. add template english_quotes.json

```json
{
    "prompt": "{quote} ->: {tags}"
}
```

2. update prompter.py and add new generate_prompt method

```python
def generate_quote_prompt(self, quote: str, tags: Union[None, list]=None,) -> str:
    tags = str(tags)
    res = self.template["prompt"].format(
        quote=quote, tags=tags
    )
    if self._verbose:
        print(res)
    return res
```

3. update generate_and_tokenize_prompt method

```python
def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_quote_prompt(
        data_point["quote"], data_point["tags"]
    )
    user_prompt = prompter.generate_quote_prompt(
        data_point["quote"], data_point["tags"]
    )
```

4. choose prompt `english_quotes` to train

```bash
python ./quotes_qlora_finetuning_cpu.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "./english_quotes" \
    --output_dir "./ipex-llm-qlora-alpaca" \
    --prompt_template_name "english_quotes"
```

### Guide to finetuning QLoRA using different models

Make sure you fully understand the entire finetune process and the model is the latest version.
Using [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main) as an example:

1. Update the Tokenizer first. Because the base example is for llama model.

```bash
from transformers import LlamaTokenizer
AutoTokenizer.from_pretrained(base_model)
```

2. Maybe some models need to add `trust_remote_code=True` in from_pretrained model and tokenizer

```
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, xxxxx, trust_remote_code=True)
```

3. Modify the `target_modules` according to the model you need to train, you can refer to [here](https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models/76779946#76779946).
   Or just search for the recommended training target modules.

```bash
lora_target_modules: List[str] = ["W_pack"]
```

4. Maybe need to change the `tokenizer.pad_token_id = tokenizer.eod_id` (Qwen)
5. (Only for baichuan) According to this [issue](https://github.com/baichuan-inc/Baichuan2/issues/204#issuecomment-1774372008),
   need to modify the [tokenization_baichuan.py](https://huggingface.co/baichuan-inc/Baichuan-7B/blob/main/tokenization_baichuan.py#L74) to fix issue.
6. finetune as normal
7. Using the [export_merged_model.py](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/QLoRA/export_merged_model.py) to merge. But also need to update tokenizer and model to ensure successful merge weight.

```bash
from transformers import AutoTokenizer  # noqa: F402
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model,trust_remote_code=True)
```

### 4. Finetuning in docker and multiple nodes (k8s)

If you want to run multi-process fine-tuning, or do not want to manually install the above dependencies, we provide a docker solution to quickly start a one-container finetuning. Please refer to [here](https://github.com/intel-analytics/ipex-llm/tree/main/docker/llm/finetune/qlora/cpu/docker#fine-tune-llm-with-ipex-llm-container).

Moreover, for users with multiple CPU server resources e.g. Xeon series like SPR and ICX, we give a k8s distributed solution, where machines and processor sockets are allowed to collaborate by one click easily. Please refer to [here](https://github.com/intel-analytics/ipex-llm/blob/main/docker/llm/finetune/qlora/cpu/kubernetes/README.md) for how to run QLoRA on k8s.
