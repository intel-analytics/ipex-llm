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
from transformers import LlamaTokenizer

from transformers import BitsAndBytesConfig
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse
from bigdl.llm.utils.isa_checker import ISAChecker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="Abirate/english_quotes")

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data = load_dataset(dataset_path)
    def merge(row):
        row['prediction'] = row['quote'] + ' ->: ' + str(row['tags'])
        return row
    data['train'] = data['train'].map(merge)
    # use the max_length to reduce memory usage, should be adjusted by different datasets
    data = data.map(lambda samples: tokenizer(samples["prediction"], max_length=256), batched=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="int4",  # nf4 not supported on cpu yet
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config, )

    # below is also supported
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              # nf4 not supported on cpu yet
    #                                              load_in_low_bit="sym_int4",
    #                                              optimize_model=False,
    #                                              torch_dtype=torch.bfloat16,
    #                                              modules_to_not_convert=["lm_head"], )

    model = model.to('cpu')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    # To avoid only one core is used on client CPU
    isa_checker = ISAChecker()
    bf16_flag = isa_checker.check_avx512()
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-4,
            save_steps=100,
            bf16=bf16_flag,
            logging_steps=20,
            output_dir="outputs",
            optim="adamw_hf",  # paged_adamw_8bit is not supported yet
            # gradient_checkpointing=True, # can further reduce memory but slower
        ),
        # Inputs are dynamically padded to the maximum length of a batch
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
