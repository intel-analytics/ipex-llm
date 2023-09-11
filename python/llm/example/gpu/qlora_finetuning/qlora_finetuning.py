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
from bigdl.llm.transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import intel_extension_for_pytorch as ipex
from peft import prepare_model_for_kbit_training
from bigdl.llm.transformers.qlora import TrainingArguments

from datasets import load_dataset

if __name__ == "__main__":

    model_path = "/mnt/disk1/models/Llama-2-13b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data = load_dataset("/home/arda/yang/BigDL/python/llm/example/gpu/qlora_finetuning/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                load_in_4bit=True,
                                                optimize_model=False,
                                                trust_remote_code=True)
    model = model.to('xpu')
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-4,
            fp16=False,
            logging_steps=20,
            output_dir="outputs",
            optim="adamw_hf", # we currently do not have paged_adamw_8bit
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
