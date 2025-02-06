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
# https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
#
# Copyright [yyyy] [name of copyright owner]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import torch

from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from ipex_llm.transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
import argparse


def chatml_format(example):
    # Format system
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune a Mistral-7b model with DPO')
    parser.add_argument('--repo-id-or-model-path', type=str, default="teknium/OpenHermes-2.5-Mistral-7B",
                        help='The huggingface repo id for the Mistral (e.g. `teknium/OpenHermes-2.5-Mistral-7B`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="Intel/orca_dpo_pairs")
    parser.add_argument('--output-path', type=str, default="outputs")
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Whether to enable gradient checkpointing to save memory at the expense of slower backward pass.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    output_path = args.output_path
    gradient_checkpointing = args.gradient_checkpointing

    # Load dataset
    dataset = load_dataset(dataset_path)['train']

    # Save columns
    original_columns = dataset.column_names

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Format dataset
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config, )

    # below is also supported
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              load_in_low_bit="nf4",
    #                                              optimize_model=False,
    #                                              torch_dtype=torch.bfloat16,
    #                                              modules_to_not_convert=["lm_head"],)

    model = model.to('xpu')
    # Prepare a IPEX-LLM compatible Peft model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    # Reference model, same as the main one
    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     load_in_low_bit="nf4",
                                                     optimize_model=False,
                                                     torch_dtype=torch.bfloat16,
                                                     modules_to_not_convert=["lm_head"],)
    ref_model = ref_model.to('xpu')

    # Training arguments
    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
        max_steps=200,
        save_strategy="no",
        logging_steps=1,
        output_dir=output_path,
        # optim="paged_adamw_32bit", # "paged_adamw_32bit" is not supported yet
        optim="adamw_hf",
        warmup_steps=100,
        bf16=True,
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    # Save artifacts
    dpo_trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
