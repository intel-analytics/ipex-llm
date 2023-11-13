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
from torch.nn import functional as F
import fire
from typing import List

import os

import transformers
from transformers import LlamaTokenizer

from peft import LoraConfig
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse

def train(
    # model/data params
    base_model: str = "./model/Llama-2-7b-chat-hf",  # the only required argument
    data_path: str = "./data/english_quotes",
    output_dir: str = "./output-mpi",
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    steps: int = 100,
    learning_rate: float = 2e-4,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
    ],
    bf16: bool = True,
):
    print(
        f"Training model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"steps: {steps}\n"
        f"learning_rate: {learning_rate}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
    )
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    pmi_world_size = int(os.environ.get('PMI_SIZE', -1))
    if pmi_world_size > 0:
        os.environ['WORLD_SIZE'] = str(pmi_world_size)
    else:
        os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"world_size: {world_size}!!")
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model_path = base_model
    dataset_path = data_path
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data = load_dataset(dataset_path)

    def merge(row):
        if row["input"]:
            row['prediction']=f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{row["instruction"]}

### Input:
{row["input"]}

### Response:
{row["output"]}"""
        else:
            row['prediction']=f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{row["instruction"]}

### Response:
{row["output"]}"""
        return row


    data['train'] = data['train'].map(merge)
    data = data.map(lambda samples: tokenizer(samples["prediction"]), batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_low_bit="sym_int4",
                                                 optimize_model=False,
                                                 torch_dtype=torch.float16,
                                                 modules_to_not_convert=["lm_head"], )
    model = model.to('cpu')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            #warmup_steps=20,
            max_steps=steps,
            #num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            save_steps=100,
            bf16=True,
            logging_steps=100,
            output_dir=output_dir,
            optim="adamw_hf",  # paged_adamw_8bit is not supported yet
            # gradient_checkpointing=True, # can further reduce memory but slower
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
if __name__ == "__main__":
    fire.Fire(train)
