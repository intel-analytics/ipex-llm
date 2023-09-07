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

from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
import numpy as np
from datetime import date

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
benchmark_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(benchmark_util_path)
from benchmark_util import BenchmarkWrapper
from bigdl.llm.utils.common.log4Error import invalidInputError

results = []


def run_model(repo_id, test_api, in_out_pairs, local_model_hub=None, warm_up=1, num_trials=3):
    # TODO: make a parameter
    if test_api == 'transformer_int4':
        result = run_transformer_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials)
        for in_out_pair in in_out_pairs:
            results.append([repo_id,
                            np.mean(result[in_out_pair], axis=0)[0],
                            np.mean(result[in_out_pair], axis=0)[1],
                            np.mean(result[in_out_pair], axis=0)[2],
                            in_out_pair])
    elif test_api == 'native_int4':
        run_native_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials)


def run_native_int4(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials):
    if local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        model_path = local_model_hub + os.path.sep + repo_model_name
    else:
        model_path = repo_id
    from bigdl.llm.transformers import BigdlNativeForCausalLM
    from bigdl.llm import llm_convert
    if "chatglm" in repo_id.lower():
        family = "chatglm"
    elif "llama" in repo_id.lower():
        family = "llama"
    else:
        invalidInputError(False, "Model family unknown: " + repo_id)

    bigdl_llm_path = llm_convert(model=model_path,
                                 outfile="./", outtype='int4', model_family=family)
    for in_out in in_out_pairs:
        in_out_len = in_out.split("-")
        in_len = int(in_out_len[0])
        out_len = int(in_out_len[1])
        input_str = open(f"prompt/{in_len}.txt", 'r').read()
        # As different tokenizer has different encodings,
        # slice the input_ids to ensure the prompt length is required length.
        n_ctx = in_len + out_len if in_len + out_len > 512 else 512
        for i in range(num_trials + warm_up):
            model = BigdlNativeForCausalLM.from_pretrained(bigdl_llm_path, model_family=family, n_ctx=n_ctx)
            input_ids = model.tokenize(input_str)
            input_ids = input_ids[:in_len]
            true_input = model.batch_decode(input_ids)
            st = time.perf_counter()
            output = model(true_input, max_tokens=out_len)
            end = time.perf_counter()
            print("model generate cost: " + str(end - st))
            print(output)

    os.remove(bigdl_llm_path)


def run_transformer_int4(repo_id,
                         local_model_hub,
                         in_out_pairs,
                         warm_up,
                         num_trials):
    if local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        model_path = local_model_hub + "/" + repo_model_name
    else:
        model_path = repo_id
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in ['THUDM/chatglm-6b', 'THUDM/chatglm2-6b']:
        model = AutoModel.from_pretrained(model_path, load_in_4bit=True, trust_remote_code=True, torch_dtype='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    end = time.perf_counter()
    print(">> loading of model costs {}s".format(end - st))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            input_str = open(f"prompt/{in_len}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len)
                end = time.perf_counter()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                print(output[0])
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time])
    return result


if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')
    today = date.today()
    
    import pandas as pd
    for api in conf.test_api:
        for model in conf.repo_id:
            run_model(model, api, conf['in_out_pairs'], conf['local_model_hub'], conf['warm_up'], conf['num_trials'])
        df = pd.DataFrame(results, columns=['model', '1st token avg latency (s)', '2+ avg latency (s/token)', 'encoder time (s)', 'input/output tokens'])
        df.to_csv(f'{current_dir}/{api}-results-{today}.csv')