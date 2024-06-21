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
# Some parts of this file is adapted from
# https://github.com/tloen/alpaca-lora/blob/main/finetune.py
#
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import accelerate

from transformers import AutoTokenizer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

current_dir = os.path.dirname(os.path.realpath(__file__))
common_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(common_util_path)
from common.utils import Prompter, get_int_from_env, wandb_check, get_train_val_data

from transformers import BitsAndBytesConfig
from ipex_llm.transformers import AutoModelForCausalLM
# import them from ipex_llm.transformers.qlora to get a IPEX-LLM compatible Peft model
from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training,\
    LoraConfig
from ipex_llm.utils.common import invalidInputError

local_rank = get_int_from_env(["LOCAL_RANK","MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE","PMI_SIZE"], "1")
port = get_int_from_env(["MASTER_PORT"], 29500)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["RANK"] = str(local_rank)
os.environ["MASTER_PORT"] = str(port)

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument, default to be "meta-llama/Llama-2-7b-hf"
    saved_low_bit_model: str = None,  # optional, the path to the saved model with ipex-llm low-bit optimization
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./bigdl-qlora-alpaca",
    # training hyperparams
    bf16: bool = True,  # default to bf16
    batch_size: int = 128,
    micro_batch_size: int = 2,  # default to be 2, limited by GPU memory
    num_epochs: int = 3,
    learning_rate: float = 3e-5,  # default to be 3e-5 to avoid divergence
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    gradient_checkpointing: bool = False,
    deepspeed: str = None,
    training_mode: str = "lora",
    deepspeed_zero3: bool = False,
    save_checkpoint: bool = True,
):
    invalidInputError(training_mode == "lora",
                      f"This example is for lora training mode, but got training_mode={training_mode}.")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
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
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"training_mode: {training_mode}\n"
            f"deepspeed_zero3: {deepspeed_zero3}\n"
            f"save_checkpoint: {save_checkpoint}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = wandb_check(wandb_project, wandb_watch, wandb_log_model)

    if saved_low_bit_model is not None:
        # Load the low bit optimized model if provide the saved path
        model = AutoModelForCausalLM.load_low_bit(
            saved_low_bit_model,
            optimize_model=False,
            torch_dtype=torch.bfloat16,
            modules_to_not_convert=["lm_head"],
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_low_bit="bf16",
            optimize_model=False,
            torch_dtype=torch.bfloat16,
             modules_to_not_convert=["lm_head"],
            trust_remote_code=True,
        )

    if deepspeed_zero3:
        deepspeed = deepspeed if deepspeed is not None else "./deepspeed_zero3_config.json"
    else:
        print(f"Model loaded on rank {os.environ.get('LOCAL_RANK')}")
        model = model.to(f'xpu:{os.environ.get("LOCAL_RANK", 0)}')
        print(f"Model moved to rank {os.environ.get('LOCAL_RANK')}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"Tokenizer loaded on rank {os.environ.get('LOCAL_RANK')}")

    # For Llama family
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(model)

    # Prepare a IPEX-LLM compatible Peft model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        training_mode=training_mode,
    )
    print(f"Lora Config: {config}")
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data, val_data = get_train_val_data(data, tokenizer, prompter, train_on_inputs,
                                              add_eos_token, cutoff_len, val_set_size, seed=42)

    # Unused
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

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
            bf16=True,  # ensure training more stable
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps" if save_checkpoint else "no",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=True if val_set_size > 0 and save_checkpoint else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            gradient_checkpointing=gradient_checkpointing,
            ddp_backend="ccl",
            deepspeed=deepspeed,
            save_safetensors=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
