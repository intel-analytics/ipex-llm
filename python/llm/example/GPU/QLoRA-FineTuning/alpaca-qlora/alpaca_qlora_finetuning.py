#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import os
os.environ["ACCELERATE_USE_IPEX"] = "1"
os.environ["ACCELERATE_USE_XPU"] = "1"

import transformers
from transformers import LlamaTokenizer
from peft import LoraConfig
import intel_extension_for_pytorch as ipex
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.qlora_utils import Prompter
from datasets import load_dataset
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Alpaca QLoRA finetuning for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned")
    parser.add_argument('--output-dir', type=str, default="./bigdl-qlora-alpaca",
                        help='The path to save the adapter.')

    args = parser.parse_args()
    base_model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    output_dir = args.output_dir

    # training hyperparams
    batch_size = 128
    micro_batch_size = 2
    num_epochs = 3
    learning_rate = 3e-5
    cutoff_len = 256
    val_set_size = 2000

    # lora hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                           "up_proj", "down_proj", "gate_proj"]

    # llm hyperparams
    train_on_inputs = True  # if False, masks out inputs in loss
    add_eos_token = False
    group_by_length = False  # faster, but produces an odd training loss curve

    prompt_template_name = "alpaca"  # The prompt template to use, will default to alpaca.
    gradient_checkpointing = False

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 load_in_low_bit="nf4",
                                                 optimize_model=False,
                                                 torch_dtype=torch.bfloat16,
                                                 # device_map=device_map,
                                                 modules_to_not_convert=["lm_head"])
    
    print(f"Model loaded on rank {os.environ.get('LOCAL_RANK')}")
    model = model.to(f'xpu:{os.environ.get("LOCAL_RANK", 0)}')
    print(f"Model moved to rank {os.environ.get('LOCAL_RANK')}")
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"Tokenizer loaded on rank {os.environ.get('LOCAL_RANK')}")
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=dataset_path)
    else:
        data = load_dataset(dataset_path)
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_ratio=0.03,
            # warmup_steps=100,
            max_grad_norm=0.3,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            gradient_checkpointing=gradient_checkpointing,
            ddp_backend="ccl",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")
