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
# https://github.com/openvino-dev-samples/Qwen2.openvino/blob/main/convert.py

from transformers import AutoTokenizer, LlamaTokenizer
from optimum.intel import OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM

import os
import omegaconf
from pathlib import Path
import argparse
import warnings

from run import LLAMA_IDS, get_model_path

current_dir = os.path.dirname(os.path.realpath(__file__))

def save_model_to_openvino(repo_id,
                           local_model_hub,
                           low_bit,
                           group_size,
                           ):
    model_path = get_model_path(repo_id, local_model_hub)

    ir_repo_id = (repo_id.split(
        "/")[1] + '-ov-' + low_bit + '-' +str(group_size))

    if local_model_hub:
        repo_model_name = repo_id.split(
        "/")[1] + '-ov-' + low_bit + '-' +str(group_size)
        ir_model_path = local_model_hub + os.path.sep + repo_model_name
        ir_model_path = Path(ir_model_path)
    else:
        ir_model_path = Path(ir_repo_id)

    if not ir_model_path.exists():
        os.mkdir(ir_model_path)

    compression_configs = {
        "sym": True,
        "group_size": group_size,
        "ratio": 1.0,
    }

    print("====Exporting IR=====")
    if low_bit == "sym_int4":
        compression_configs['sym'] = True
        ov_model = OVModelForCausalLM.from_pretrained(model_path, export=True, 
                                                      trust_remote_code=True,
                                                      compile=False, quantization_config=OVWeightQuantizationConfig(
                                                      bits=4, **compression_configs)).eval()
    elif low_bit == "asym_int4":
        compression_configs['sym'] = False
        ov_model = OVModelForCausalLM.from_pretrained(model_path, export=True, 
                                                      trust_remote_code=True,
                                                      compile=False, quantization_config=OVWeightQuantizationConfig(
                                                      bits=4, **compression_configs)).eval()

    print("====Saving IR=====")
    ov_model.save_pretrained(ir_model_path)

    print("====Exporting tokenizer=====")
    if repo_id in LLAMA_IDS:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)
    tokenizer.save_pretrained(ir_model_path)

    print("====Exporting IR tokenizer=====")
    from optimum.exporters.openvino.convert import export_tokenizer
    export_tokenizer(tokenizer, ir_model_path)
    print("====Finished=====")
    del ov_model
    del model_path

if __name__ == '__main__':
    supported_precision = ["sym_int4", "asym_int4"]

    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    if conf['low_bit'] in supported_precision:
        for model in conf.repo_id:
            save_model_to_openvino(repo_id=model,
                                local_model_hub=conf['local_model_hub'],
                                low_bit=conf['low_bit'],
                                group_size=conf['group_size'],)
    else:
        warnings.warn("The precision selected is not supported in our all-in-one benchmark.")
