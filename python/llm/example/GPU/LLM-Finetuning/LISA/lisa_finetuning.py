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

import os
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import accelerate
from ipex_llm.transformers.lisa import DynamicLayerActivationCallback
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.realpath(__file__))
common_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(common_util_path)
from common.utils import Prompter, get_train_val_data

from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.utils.common import invalidInputError

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument, default to be "meta-llama/Llama-2-7b-hf"
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./ipex-llm-lisa-alpaca",
    # training hyperparams
    bf16: bool = True,  # default to bf16
    batch_size: int = 128,
    micro_batch_size: int = 8,  # default to be 8, limited by GPU memory
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    gradient_checkpointing: bool = False,
    deepspeed: str = None,
    training_mode: str = "lisa",
    lisa_activated_layers: int = 1,
    lisa_interval_steps: int = 20,
):
    invalidInputError(training_mode == "lisa",
                      f"This example is for lisa training mode, but got training_mode={training_mode}.")
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
        f"training_mode: {training_mode}\n"
        f"lisa_activated_layers: {lisa_activated_layers}\n"
        f"lisa_interval_steps: {lisa_interval_steps}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_low_bit="bf16",
        optimize_model=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        modules_to_not_convert=["lm_head"],     # avoid optimize lm_head
    )

    model = model.to("xpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    print(model)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_data, val_data = get_train_val_data(data, tokenizer, prompter, train_on_inputs,
                                              add_eos_token, cutoff_len, val_set_size, seed=42)

    trainer_callbacks = []

    # Instantiate the callback
    dynamic_layer_activation_callback = DynamicLayerActivationCallback(
        n_layers=lisa_activated_layers,                     # Number of layers to activate
        interval_steps = lisa_interval_steps,               # Step interval to update active layers
        model = model
    )
    trainer_callbacks.append(dynamic_layer_activation_callback)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=0.3,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            bf16=bf16,  # ensure training more stable
            logging_steps=10,
            optim="adamw_hf",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            gradient_checkpointing=gradient_checkpointing,
            deepspeed=deepspeed,
            save_safetensors=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=trainer_callbacks
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model.save_pretrained(output_dir)
    trainer.save_model()


if __name__ == "__main__":
    fire.Fire(train)
