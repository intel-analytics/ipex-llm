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
#
# Some parts of this file is adapted from https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
#
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

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
import transformers


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)


def wandb_check(wandb_project, wandb_watch, wandb_log_model):
    """Check if wandb related parameter passed or if set within environ"""
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    return use_wandb


def get_train_val_data(data, tokenizer, prompter, train_on_inputs,
                       add_eos_token, cutoff_len, val_set_size, seed=42):
    """Data processing to get train data and val data"""
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

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
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
    return train_data, val_data


def merge_adapter(base_model, tokenizer, adapter_path, output_path):
    """Merge the adapter into the original model and save"""
    import torch
    from ipex_llm.transformers.qlora import PeftModel, LoraConfig
    from ipex_llm.transformers import AutoModelForCausalLM
    from ipex_llm.transformers.low_bit_linear import get_block_size
    import tempfile
    import shutil

    lora_config = LoraConfig.from_json_file(os.path.join(adapter_path, "adapter_config.json"))
    training_mode = lora_config.get("training_mode", "qlora")
    qa_lora = training_mode == "qalora"

    temp_dir = None
    if qa_lora:
        # Convert the qa-lora adapter to the correct shapes
        # The default 4-bit format for qa_lora is sym_int4
        block_size = get_block_size("sym_int4")
        temp_dir = tempfile.TemporaryDirectory()
        tmpdirname = os.path.join(temp_dir.name, "adapter")
        try:
            shutil.copytree(adapter_path, tmpdirname)
        except Exception as e:
            print(f"Failed to copy adapter dir, error: {e}")
        mid_lora_path = os.path.join(tmpdirname, "adapter_model.bin")

        adapter_path = os.path.join(adapter_path, "adapter_model.bin")

        lora = torch.load(adapter_path, map_location='cpu')
        # Get lora_a names
        tmp_keys = [key for key in lora.keys() if 'lora_A' in key]

        for tmp_key in tmp_keys:
            lora_a = lora[tmp_key] / block_size
            lora[tmp_key] = torch.repeat_interleave(lora_a, block_size, dim=1)

        torch.save(lora, mid_lora_path)
        adapter_path = tmpdirname

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_low_bit="nf4", # should load the orignal model
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )

        # merge weights - new merging method from peft
        lora_model = lora_model.merge_and_unload()

        lora_model.train(False)

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }

        base_model.save_pretrained(output_path, state_dict=deloreanized_sd)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Failed to merge the adapter, error: {e}.")
    finally:
        if qa_lora and temp_dir:
           temp_dir.cleanup()
