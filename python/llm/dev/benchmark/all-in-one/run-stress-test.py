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


# this code is copied from llama2 example test, and added performance test
import torch
import time
import gc
import traceback
import threading

import numpy as np
from datetime import date

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
import sys
from ipex_llm.utils import BenchmarkWrapper
from ipex_llm.utils.common.log4Error import invalidInputError

LLAMA_IDS = ['meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-13b-chat-hf',
             'meta-llama/Llama-2-70b-chat-hf','decapoda-research/llama-7b-hf',
             'decapoda-research/llama-65b-hf','lmsys/vicuna-7b-v1.5',
             'lmsys/vicuna-13b-v1.3','project-baize/merged-baize-30b']

CHATGLM_IDS = ['THUDM/chatglm-6b', 'THUDM/chatglm2-6b', 'THUDM/chatglm3-6b']

LLAVA_IDS = ['liuhaotian/llava-v1.5-7b']

results = []
excludes = []

def run_model(repo_id, test_api, in_out_pairs, local_model_hub=None, warm_up=1, num_trials=3, num_beams=1, low_bit='sym_int4', cpu_embedding=False):
    # TODO: make a parameter
    result= {}
    if test_api == 'transformer_int4':
        result = run_transformer_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit)
    elif test_api == 'transformer_int4_gpu':
        result = run_transformer_int4_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit)


    for in_out_pair in in_out_pairs:
        if result and result[in_out_pair]:
            results.append([repo_id,
                            round(np.mean(result[in_out_pair], axis=0)[0]*1000.0, 2),
                            round(np.mean(result[in_out_pair], axis=0)[1]*1000.0, 2),
                            round(np.mean(result[in_out_pair], axis=0)[2]*1000.0, 2),
                            in_out_pair,
                            f'{int(np.mean(result[in_out_pair], axis=0)[3])}' +
                            f'-{int(np.mean(result[in_out_pair], axis=0)[4])}',
                            num_beams,
                            low_bit,
                            cpu_embedding if 'win' in test_api else 'N/A',
                            result[in_out_pair][-1][5] if 'int4_gpu' in test_api else 'N/A']) # currently only peak mem for win gpu is caught here

def get_model_path(repo_id, local_model_hub):
    if local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        local_model_path = local_model_hub + os.path.sep + repo_model_name
        invalidInputError(os.path.isdir(local_model_path),
                          local_model_path + " not exists!, Please check your models' folder.")
        return local_model_path
    else:
        return repo_id

def run_transformer_int4(repo_id,
                         local_model_hub,
                         in_out_pairs,
                         warm_up,
                         num_trials,
                         num_beams,
                         low_bit):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True, torch_dtype='auto').eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    end = time.perf_counter()
    print(">> loading of model costs {}s".format(end - st))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            i = 0
            with open("prompt/stress_test.txt", 'r') as file:
                for input_str in file:
                    # As different tokenizer has different encodings,
                    # slice the input_ids to ensure the prompt length is required length.
                    input_ids = tokenizer.encode(input_str, return_tensors="pt")
                    input_ids = input_ids[:, :in_len]
                    true_str = tokenizer.batch_decode(input_ids)[0]
                    input_ids = tokenizer.encode(true_str, return_tensors="pt")
                    actual_in_len = input_ids.shape[1]
                    result[in_out] = []
                    st = time.perf_counter()
                    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                num_beams=num_beams)
                    end = time.perf_counter()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                            actual_in_len, actual_out_len])
                    i += 1
                    if i >= warm_up+num_trials:
                        break

    return result

def run_transformer_int4_gpu(repo_id,
                             local_model_hub,
                             in_out_pairs,
                             warm_up,
                             num_trials,
                             num_beams,
                             low_bit):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer
    import intel_extension_for_pytorch as ipex
    reserved_mem_list = []
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    print(">> loading of model costs {}s".format(end - st))
    reserved_mem_list.append(torch.xpu.memory.memory_reserved()/(1024**3))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            i = 0
            with open("prompt/stress_test.txt", 'r') as file:
                for input_str in file:
                    # As different tokenizer has different encodings,
                    # slice the input_ids to ensure the prompt length is required length.
                    input_ids = tokenizer.encode(input_str, return_tensors="pt")
                    input_ids = input_ids[:, :in_len]
                    true_str = tokenizer.batch_decode(input_ids)[0]
                    input_ids = tokenizer.encode(true_str, return_tensors="pt").to('xpu')
                    actual_in_len = input_ids.shape[1]
                    result[in_out] = []
                    st = time.perf_counter()
                    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                num_beams=num_beams)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    reserved_mem_list.append(torch.xpu.memory.memory_reserved()/(1024**3))
                    gpu_peak_mem = max(reserved_mem_list) # always keep the peak gpu mem at current stage
                    output_ids = output_ids.cpu()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                            actual_in_len, actual_out_len, gpu_peak_mem])
                    i += 1
                    if i >= warm_up+num_trials:
                        break
    model.to('cpu')
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    del model
    gc.collect()
    return result

if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')
    today = date.today()
    if 'exclude' in conf:
        excludes = conf['exclude']

    import pandas as pd
    for api in conf.test_api:
        for model in conf.repo_id:
            in_out_pairs = conf['in_out_pairs'].copy()
            if excludes:
                for in_out in conf['in_out_pairs']:
                    model_id_input = model + ':' + in_out.split('-')[0]
                    if model_id_input in excludes:
                        in_out_pairs.remove(in_out)
            run_model(model, api, in_out_pairs, conf['local_model_hub'], conf['warm_up'], conf['num_trials'], conf['num_beams'],
                      conf['low_bit'], conf['cpu_embedding'])
        df = pd.DataFrame(results, columns=['model', '1st token avg latency (ms)', '2+ avg latency (ms/token)', 'encoder time (ms)',
                                            'input/output tokens', 'actual input/output tokens', 'num_beams', 'low_bit', 'cpu_embedding',
                                            'peak mem (GB)'])

        df.to_csv(f'{current_dir}/{api}-results-{today}.csv')
        results = []
