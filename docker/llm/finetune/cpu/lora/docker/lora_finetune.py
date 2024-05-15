import os
import sys
from typing import List
import time

import fire
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset


"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

# Catch when user should re-install transformers library
# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"  # noqa: E501

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from transformers import BitsAndBytesConfig


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "./alpaca_data_cleaned.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "", # options: false | gradients | all
    wandb_log_model: str = "", # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    use_ipex: bool = False,
    bf16: bool = False,
    no_cuda: bool=True,
    xpu_backend: str="ccl"
):
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
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"use_ipex: {use_ipex}\n"
        f"bf16: {bf16}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
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
    local_rank = 0
    if ddp:
        os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
        os.environ['LOCAL_RANK'] = str(os.environ.get('PMI_RANK', 0))
        local_rank = str(os.environ.get('PMI_RANK', 0))
        print("PMI_RANK(local_rank): " + local_rank)
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or \
                ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ['WANDB_PROJECT'] = wandb_project
    if len(wandb_watch) > 0:
        os.environ['WANDB_WATCH'] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ['WANDB_LOG_MODEL'] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

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
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        print("[INFO] spliting and shuffling dataset...")
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        print("[INFO] shuffling and tokenizing train data...")
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        print("[INFO] shuffling and tokenizing test data...")
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    print("[INFO] begining the training of transformers...")

    args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="steps",
            local_rank=local_rank,
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            xpu_backend=xpu_backend,
            no_cuda=no_cuda
    )

    print(
        f"[INFO] Process rank: {args.local_rank}, device: {args.device}"
        + f"distributed training: {args.parallel_mode.value == 'distributed'}"
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end = time.time()
    print("training time is: ", end - start)

    if int(os.environ.get("PMI_RANK", -1)) == 0:
        model.save_pretrained(output_dir)
    elif int(os.environ.get("PMI_RANK", -1)) == -1:
        model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
