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

from transformers import AutoTokenizer
from optimum.intel import OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM

import os
import omegaconf
from pathlib import Path
import argparse

from run import LLAMA_IDS, CHATGLM_IDS, LLAVA_IDS, PHI3VISION_IDS, QWENVL_IDS, get_model_path

current_dir = os.path.dirname(os.path.realpath(__file__))

def save_model_to_openvino(repo_id,
                           local_model_hub,
                           low_bit,
                           output,
                           group_size,
                           modelscope,
                           ):
    model_path = get_model_path(repo_id, local_model_hub)

    ir_model_path = Path(repo_id.split(
        "/")[1] + '-ov-' + str(group_size)) if output is None else Path(output)

    if not ir_model_path.exists():
        os.mkdir(ir_model_path)

    compression_configs = {
        "sym": True,
        "group_size": group_size,
        "ratio": 1.0,
    }
    if modelscope:
        from modelscope import snapshot_download

        print("====Downloading model from ModelScope=====")
        model_path = snapshot_download(repo_id, cache_dir='./')

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
    elif args.precision == "int8":
        ov_model = OVModelForCausalLM.from_pretrained(model_path, export=True,
                                                      trust_remote_code=True,
                                                      compile=False, load_in_8bit=True).eval()
    else:
        ov_model = OVModelForCausalLM.from_pretrained(model_path, export=True,
                                                      trust_remote_code=True,
                                                      compile=False, load_in_8bit=False).eval()

    print("====Saving IR=====")
    ov_model.save_pretrained(ir_model_path)

    print("====Exporting tokenizer=====")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path)
    tokenizer.save_pretrained(ir_model_path)

    print("====Exporting IR tokenizer=====")
    from optimum.exporters.openvino.convert import export_tokenizer
    export_tokenizer(tokenizer, ir_model_path)
    print("====Finished=====")
    del ov_model
    del model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        default='Qwen/Qwen1.5-0.5B-Chat',
                        required=False,
                        type=str,
                        help='orignal model path')
    parser.add_argument('-p',
                        '--precision',
                        required=False,
                        default="int4",
                        type=str,
                        choices=["fp16", "int8", "int4"],
                        help='fp16, int8 or int4')
    parser.add_argument('-o',
                        '--output',
                        required=False,
                        type=str,
                        help='path to save the ir model')
    parser.add_argument('-g',
                        '--group_size',
                        required=False,
                        default=64,
                        type=int,
                        help='group size decided to use'
    )
    parser.add_argument('-ms',
                        '--modelscope',
                        action='store_true',
                        help='download model from Model Scope')
    args = parser.parse_args()

    output = args.output
    group_size = args.group_size
    modelscope = args.modelscope

    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    for model in conf.repo_id:
        save_model_to_openvino(repo_id=model,
                              local_model_hub=conf['local_model_hub'],
                              low_bit=conf['low_bit'],
                              output=output,
                              group_size=group_size,
                              modelscope=modelscope)
