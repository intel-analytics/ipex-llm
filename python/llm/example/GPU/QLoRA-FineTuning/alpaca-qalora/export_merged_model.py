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
# This file is adapted from https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
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
import os
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from bigdl.llm.transformers.qlora import PeftModel


lora_path = "/home/arda/yina/checkpoint-1100"
base_path = "/mnt/disk1/models/Llama-2-7b-chat-hf"

mid_path = "/home/arda/yina/qalora_adapter/"
mid_lora_path = os.path.join(mid_path, "adapter_model.bin")

lora_config = LoraConfig.from_json_file(os.path.join(lora_path, "adapter_config.json"))
lora_scale = lora_config["lora_alpha"] / lora_config["r"]

lora_path = os.path.join(lora_path, "adapter_model.bin")

# # model = torch.load(base_path, map_location='cpu')
base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        # load_in_low_bit="nf4", # should load the orignal model
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
lora = torch.load(lora_path, map_location='cpu')
tmp_keys = [key[17:-14] for key in lora.keys() if 'lora_A' in key]

for tmp_key in tmp_keys:
    a = lora['base_model.model.'+tmp_key+'.lora_A.weight']
    # b = lora['base_model.model.'+tmp_key+'.lora_B.weight']
    lora['base_model.model.'+tmp_key+'.lora_A.weight'] = torch.repeat_interleave(a, 64, dim=1) * lora_scale / 64
    # lora['base_model.model.'+tmp_key+'.lora_B.weight'] = torch.repeat_interleave(b, 64, dim=0) * lora_scale / 64

torch.save(lora, mid_lora_path)

# lora2 = torch.load(mid_lora_path, map_location='cpu')
# for tmp_key in tmp_keys:
#     a = lora['base_model.model.'+tmp_key+'.lora_A.weight']
#     b = lora['base_model.model.'+tmp_key+'.lora_B.weight']
#     print(f"{tmp_key}, lora a size: {a.shape}, lora b size: {b.shape}")

lora_model = PeftModel.from_pretrained(
        base_model,
        mid_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

lora_model = lora_model.merge_and_unload()

lora_model.train(False)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}
