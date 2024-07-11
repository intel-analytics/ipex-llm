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

# this code is to support converting of model in load bit
# for performance tests using load_low_bit

import omegaconf
import time
import os
import sys
import gc

from run import LLAMA_IDS, CHATGLM_IDS, LLAVA_IDS, PHI3VISION_IDS, QWENVL_IDS, get_model_path

current_dir = os.path.dirname(os.path.realpath(__file__))

def save_model_in_low_bit(repo_id,
                          local_model_hub,
                          low_bit):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAVA_IDS:
        llava_repo_dir = os.environ.get('LLAVA_REPO_DIR')
        sys.path.append(rf"{llava_repo_dir}")
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in PHI3VISION_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     _attn_implementation="eager",
                                                     modules_to_not_convert=["vision_embed_tokens"],
                                                     trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in QWENVL_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     modules_to_not_convert=['c_fc', 'out_proj'],
                                                     trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    end = time.perf_counter()
    print(">> loading of and converting of model costs {}s".format(end - st))

    model.save_low_bit(model_path+'-'+low_bit)
    tokenizer.save_pretrained(model_path+'-'+low_bit)

    del model
    gc.collect()

if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    for model in conf.repo_id:
        save_model_in_low_bit(repo_id=model,
                              local_model_hub=conf['local_model_hub'],
                              low_bit=conf['low_bit'])
