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

import transformers
from transformers import AutoTokenizer
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from ipex_llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
common_util_path = os.path.join(current_dir, '..', '..')
import sys
sys.path.append(common_util_path)
from common.utils import Prompter, get_train_val_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple example of how to qlora finetune llama2 model using bigdl-llm and TRL')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned")

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Avoid tokenizer doesn't have a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=dataset_path)
    else:
        data = load_dataset(dataset_path)

    # For illustration purpose, only use part of data to train
    data = data["train"].train_test_split(train_size=0.1, shuffle=False)

    # Data processing
    prompter = Prompter("alpaca")
    train_data, _ = get_train_val_data(data, tokenizer, prompter, train_on_inputs=True,
                                       add_eos_token=False, cutoff_len=256, val_set_size=0, seed=42)

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
    #                                             load_in_low_bit="nf4",
    #                                             optimize_model=False,
    #                                             torch_dtype=torch.bfloat16,
    #                                             modules_to_not_convert=["lm_head"],)
    model = model.to('xpu')
    # Enable gradient_checkpointing if your memory is not enough,
    # it will slowdown the training speed
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps= 1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-5,
            save_steps=100,
            bf16=True,  # bf16 is more stable in training
            logging_steps=20,
            output_dir="outputs",
            optim="adamw_hf", # paged_adamw_8bit is not supported yet
            gradient_checkpointing=True, # can further reduce memory but slower
        ),
        dataset_text_field="instruction",
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
