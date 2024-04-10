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

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import setup_chat_format
from datasets import load_dataset
import torch
import argparse

# GaLore hyperparameters
rank = 1024
update_proj_gap = 200
scale = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune transformers with IPEX-LLM GaLore')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--data-path', type=str, default="g-ronimo/oasst2_top4k_en",
                        help='Dataset path for fine-tuning')
    parser.add_argument('--output-dir', type=str, default="./ipex-llm-galore",
                        help='Path to save fine-tuned mode')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        optimize_model=True,
        use_cache = False,
        trust_remote_code=True,
    )
    model = model.to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)

    model, tokenizer = setup_chat_format(model, tokenizer)
    if tokenizer.pad_token in [None, tokenizer.eos_token]:
        tokenizer.pad_token = tokenizer.unk_token

    dataset = load_dataset(args.data_path)

    from transformers import TrainingArguments

    training_arguments = TrainingArguments(
        output_dir = f"out",
        evaluation_strategy = "steps",
        label_names = ["labels"],
        per_device_train_batch_size = 16,
        save_steps = 250,
        eval_steps = 250,
        logging_steps = 1,
        learning_rate = 1e-5,
        num_train_epochs = 3,
        lr_scheduler_type = "constant",
        gradient_checkpointing = True,
        optim = "galore_adamw_layerwise",
        optim_target_modules = ["attn", "mlp"],
        optim_args = f"rank={rank}, update_proj_gap={update_proj_gap}, scale={scale}",
    )

    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset['test'],
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template = "<|im_start|>user",
            response_template = "<|im_start|>assistant",
            tokenizer = tokenizer,
            mlm = False),
        max_seq_length = 256,
        dataset_kwargs = dict(add_special_tokens = False),
        args = training_arguments,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
