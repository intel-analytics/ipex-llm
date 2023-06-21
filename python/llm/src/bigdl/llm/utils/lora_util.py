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
# ===========================================================================
#
# This file is adapted from
# https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
#
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import torch
from peft import PeftModel, PeftConfig
from bigdl.llm.utils import invalidInputError
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer  # noqa: F402


def merge_and_export_lora_model(base_model, lora_id_or_path, output_path="./hf_ckpt"):
    config = get_tokenizer_config(base_model)
    if config.get("tokenizer_class", "AutoTokenizer") == 'LlamaTokenizer':
        # Llama tokenizer load very slow with AutoTokenizer, use LlamaTokenizer here
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    config = PeftConfig.from_pretrained(lora_id_or_path)

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_id_or_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    lora_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path

if __name__ == '__main__':
    merge_and_export_lora_model("huggyllama/llama-7b", "tloen/alpaca-lora-7b", "./hf_ckpt")
