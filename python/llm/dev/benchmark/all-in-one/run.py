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
import csv

import numpy as np
from datetime import date

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
benchmark_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(benchmark_util_path)
from benchmark_util import BenchmarkWrapper
from ipex_llm.utils.common.log4Error import invalidInputError

LLAMA_IDS = ['meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-13b-chat-hf',
             'meta-llama/Llama-2-70b-chat-hf','decapoda-research/llama-7b-hf',
             'decapoda-research/llama-65b-hf','lmsys/vicuna-7b-v1.5',
             'lmsys/vicuna-13b-v1.3','lmsys/vicuna-33b-v1.3','project-baize/merged-baize-30b']

CHATGLM_IDS = ['THUDM/chatglm-6b', 'THUDM/chatglm2-6b', 'THUDM/chatglm3-6b']

LLAVA_IDS = ['liuhaotian/llava-v1.5-7b']

results = []
excludes = []

def run_model_in_thread(model, in_out, tokenizer, result, warm_up, num_beams, input_ids, out_len, actual_in_len, num_trials, load_time):
    for i in range(num_trials + warm_up):
        st = time.perf_counter()
        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                    num_beams=num_beams)
        torch.xpu.synchronize()
        end = time.perf_counter()
        output_ids = output_ids.cpu()
        print("model generate cost: " + str(end - st))
        output = tokenizer.batch_decode(output_ids)
        print(output[0])
        torch.xpu.empty_cache()
        actual_out_len = output_ids.shape[1] - actual_in_len
        if i >= warm_up:
            result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                   actual_in_len, actual_out_len, load_time, model.peak_memory])

def run_model(repo_id, test_api, in_out_pairs, local_model_hub=None, warm_up=1, num_trials=3, num_beams=1, low_bit='sym_int4', cpu_embedding=False, batch_size=1, streaming=False, use_fp16_torch_dtype=False, n_gpu=2):
    # TODO: make a parameter
    result= {}
    if test_api == 'transformer_int4':
        result = run_transformer_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'native_int4':
        run_native_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials)
    elif test_api == 'optimize_model':
        result = run_optimize_model(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'transformer_int4_gpu':
        result = run_transformer_int4_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size, cpu_embedding)
    elif test_api == 'transformer_int4_fp16_gpu':
        result = run_transformer_int4_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size, cpu_embedding, fp16=True)
    elif test_api == 'optimize_model_gpu':
        result = run_optimize_model_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'pytorch_autocast_bf16':
        result = run_pytorch_autocast_bf16(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'ipex_fp16_gpu':
        result = run_ipex_fp16_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == "bigdl_fp16_gpu":
        result = result = run_bigdl_fp16_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'deepspeed_transformer_int4_cpu':
        result = run_deepspeed_transformer_int4_cpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'transformer_int4_gpu_win':
        result = run_transformer_int4_gpu_win(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, cpu_embedding, batch_size, streaming)
    elif test_api == 'transformer_int4_fp16_gpu_win':
        result = run_transformer_int4_fp16_gpu_win(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, cpu_embedding, batch_size, streaming)
    elif test_api == 'transformer_int4_loadlowbit_gpu_win':
        # drop the results of the first time for better performance
        run_transformer_int4_loadlowbit_gpu_win(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, cpu_embedding, batch_size, streaming)
        result = run_transformer_int4_loadlowbit_gpu_win(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, cpu_embedding, batch_size, streaming)
    elif test_api == 'transformer_autocast_bf16':
        result = run_transformer_autocast_bf16(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'bigdl_ipex_bf16':
        result = run_bigdl_ipex_bf16(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'bigdl_ipex_int4':
        result = run_bigdl_ipex_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'bigdl_ipex_int8':
        result = run_bigdl_ipex_int8(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'deepspeed_optimize_model_gpu':
        result = run_deepspeed_optimize_model_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size, cpu_embedding)
    elif test_api == 'speculative_cpu':
        result = run_speculative_cpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'speculative_gpu':
        result = run_speculative_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)
    elif test_api == 'pipeline_parallel_gpu':
        result = run_pipeline_parallel_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size, cpu_embedding, fp16=use_fp16_torch_dtype, n_gpu=n_gpu)

    for in_out_pair in in_out_pairs:
        if result and result[in_out_pair]:
            results.append([repo_id,
                            round(np.mean(result[in_out_pair], axis=0)[0]*1000.0, 2),
                            round(np.mean(result[in_out_pair], axis=0)[1]*1000.0, 2),
                            round(np.mean(result[in_out_pair], axis=0)[2]*1000.0, 2),
                            in_out_pair,
                            batch_size,
                            f'{int(np.mean(result[in_out_pair], axis=0)[3])}' +
                            f'-{int(np.mean(result[in_out_pair], axis=0)[4])}',
                            num_beams,
                            low_bit,
                            cpu_embedding,
                            round(result[in_out_pair][-1][5], 2),
                            result[in_out_pair][-1][6] if any(keyword in test_api for keyword in ['int4_gpu', 'int4_fp16_gpu_win', 'int4_loadlowbit_gpu', 'fp16_gpu', 'deepspeed_optimize_model_gpu']) else 'N/A',
                            streaming if 'win' in test_api else 'N/A',
                            use_fp16_torch_dtype if 'pipeline_parallel_gpu' in test_api else 'N/A'],
                            ) 


def get_model_path(repo_id, local_model_hub):
    if local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        local_model_path = local_model_hub + os.path.sep + repo_model_name
        invalidInputError(os.path.isdir(local_model_path),
                          local_model_path + " not exists!, Please check your models' folder.")
        return local_model_path
    else:
        return repo_id


def run_native_int4(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials):
    model_path = get_model_path(repo_id, local_model_hub)
    from ipex_llm.transformers import BigdlNativeForCausalLM
    from ipex_llm import llm_convert
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
                         num_trials,
                         num_beams,
                         low_bit,
                         batch_size):
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
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
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
                                           actual_in_len, actual_out_len, load_time])
    return result

def run_pytorch_autocast_bf16(repo_id,
                         local_model_hub,
                         in_out_pairs,
                         warm_up,
                         num_trials,
                         num_beams,
                         batch_size):
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer

    model_path = get_model_path(repo_id, local_model_hub)
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        # TODO: need verify chatglm family run bf16.
        print("Currently pytorch do not support bfloat16 on cpu for chatglm models. Will skip it")
        return
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True)
        # Need to use LlamaTokenizer, reason please refer to issue: https://github.com/intel-analytics/BigDL/issues/8944
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)
    result = {}
    with torch.inference_mode(), torch.autocast("cpu"):
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            print("input tokens: {}".format(input_ids.shape[1]))
            for i in range(num_trials + warm_up):
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
                                           actual_in_len, actual_out_len, load_time])
    return result

def run_optimize_model(repo_id,
                       local_model_hub,
                       in_out_pairs,
                       warm_up,
                       num_trials,
                       num_beams,
                       low_bit,
                       batch_size):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
    from ipex_llm import optimize_model

    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True, trust_remote_code=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True, low_cpu_mem_usage=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
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
                                           actual_in_len, actual_out_len, load_time])
    return result


def run_transformer_int4_gpu(repo_id,
                             local_model_hub,
                             in_out_pairs,
                             warm_up,
                             num_trials,
                             num_beams,
                             low_bit,
                             batch_size,
                             cpu_embedding,
                             fp16=False):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    origin_repo_id = repo_id.replace("-4bit", "")
    if origin_repo_id in CHATGLM_IDS:
        if "4bit" in repo_id:
            model = AutoModel.load_low_bit(model_path, optimize_model=True,
                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()  
        else:
            model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                            trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cpu_embedding=cpu_embedding)
    elif origin_repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        if "4bit" in repo_id:
            model = AutoModelForCausalLM.load_low_bit(model_path, optimize_model=True,
                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        else:
            if 'starcoder' in repo_id:
                # Load starcoder-15.5b model in bf16 format to avoid CPU OOM.
                model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding, torch_dtype=torch.bfloat16).eval()
                # Convert the low-bit model back to fp32 for performance considerations.
                model = model.float()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if fp16:
        model = model.half()
        print("Convert model to half precision")
    
    model = model.to('xpu')

    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s and {}GB".format(load_time, torch.xpu.memory.memory_reserved()/(1024**3)))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192] and test_length < 8192:
                test_length = test_length * 2
            # For the sequence length not in [32, 256, 1024, 2048, 8192], it will be truncated from 8192.txt.
            test_length = min(test_length, 8192)
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            thread = threading.Thread(target=run_model_in_thread, args=(model, in_out, tokenizer, result, warm_up, num_beams, input_ids, out_len, actual_in_len, num_trials, load_time))
            thread.start()
            thread.join()

            if result[in_out]:
                first_token_latency = round(np.mean(result[in_out], axis=0)[0]*1000.0, 2)
                rest_token_latency = round(np.mean(result[in_out], axis=0)[1]*1000.0, 2)
                encoder_time = round(np.mean(result[in_out], axis=0)[2]*1000.0, 2)
                input_output_tokens = in_out
                actual_input_output_tokens = f'{int(np.mean(result[in_out], axis=0)[3])}' + f'-{int(np.mean(result[in_out], axis=0)[4])}'
                load_time = round(result[in_out][-1][5], 2)
                peak_mem = result[in_out][-1][6]
                with open(csv_name, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    file.seek(0, os.SEEK_END)
                    if file.tell() == 0:
                        csv_writer.writerow(["","model","1st token avg latency (ms)","2+ avg latency (ms/token)","encoder time (ms)","input/output tokens", "batch_size", "actual input/output tokens","num_beams","low_bit","cpu_embedding","model loading time (s)","peak mem (GB)"])
                    csv_writer.writerow(['', repo_id, first_token_latency, rest_token_latency, encoder_time, input_output_tokens, batch_size, actual_input_output_tokens, num_beams, low_bit, '', load_time, peak_mem])

    model.to('cpu')
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    del model
    gc.collect()
    return result

def run_optimize_model_gpu(repo_id,
                           local_model_hub,
                           in_out_pairs,
                           warm_up,
                           num_trials,
                           num_beams,
                           low_bit,
                           batch_size):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    from ipex_llm import optimize_model
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True,
                                          trust_remote_code=True, use_cache=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True, low_cpu_mem_usage=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True,
                                                     trust_remote_code=True, use_cache=True).eval()
        model = optimize_model(model, low_bit=low_bit)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                           actual_in_len, actual_out_len, load_time])
    del model
    torch.xpu.empty_cache()
    return result


def run_ipex_fp16_gpu(repo_id,
                      local_model_hub,
                      in_out_pairs,
                      warm_up,
                      num_trials,
                      num_beams,
                      batch_size):
    from transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half().to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half().to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half().to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                           actual_in_len, actual_out_len, load_time])
    del model
    torch.xpu.empty_cache()
    return result


def run_bigdl_fp16_gpu(repo_id,
                       local_model_hub,
                       in_out_pairs,
                       warm_up,
                       num_trials,
                       num_beams,
                       batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_cache=True,
                                          load_in_low_bit="fp16", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True,
                                                     load_in_low_bit="fp16",
                                                     torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True,
                                                     load_in_low_bit="fp16",
                                                     torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                           actual_in_len, actual_out_len, load_time, model.peak_memory])
    del model
    torch.xpu.empty_cache()
    return result

def run_deepspeed_transformer_int4_cpu(repo_id,
                         local_model_hub,
                         in_out_pairs,
                         warm_up,
                         num_trials,
                         num_beams,
                         low_bit,
                         batch_size):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
    import deepspeed
    from ipex_llm import optimize_model
    import argparse
    # parser is for deepspeed subprocesses' inline parameter
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--local_rank', type=str, default=0, help='this is automatically set when using deepspeed launcher')
    args = parser.parse_args()
    local_rank = int(os.getenv("RANK", "1"))
    if local_rank == -1:
        local_rank = args.local_rank
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    model_path = get_model_path(repo_id, local_model_hub)

    st = time.perf_counter()
    # Note: only tested cpu Llama2-7b
    # Native Huggingface transformers loading to enable deepspeed init
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Parallelize model on deepspeed
    model = deepspeed.init_inference(model, mp_size=world_size,
                                     dtype=torch.float16,
                                     replace_method="auto")

    # Apply BigDL-LLM INT4 optimization to enable BenchmarkWrapper
    # Note: only tested sym_int4
    model = optimize_model(model.module.to(f'cpu'), low_bit=low_bit)
    model = model.to(f'cpu:{local_rank}')

    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                num_beams=num_beams)
                end = time.perf_counter()
                if local_rank == 0:
                    print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                if local_rank == 0:
                    print(output[0])
                actual_out_len = output_ids.shape[1] - actual_in_len
                if i >= warm_up :
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                           actual_in_len, actual_out_len, load_time])
    return result


def run_transformer_int4_gpu_win(repo_id,
                                 local_model_hub,
                                 in_out_pairs,
                                 warm_up,
                                 num_trials,
                                 num_beams,
                                 low_bit,
                                 cpu_embedding,
                                 batch_size,
                                 streaming):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer, TextStreamer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                                     trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAVA_IDS:
        llava_repo_dir = os.environ.get('LLAVA_REPO_DIR')
        sys.path.append(rf"{llava_repo_dir}")
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s and {}GB".format(load_time, torch.xpu.memory.memory_reserved()/(1024**3)))

    model = BenchmarkWrapper(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            try:
                in_out_len = in_out.split("-")
                in_len = int(in_out_len[0])
                out_len = int(in_out_len[1])
                # As different tokenizer has different encodings,
                # in_len.txt maybe shorter than we need,
                # use much longer context to make sure input length
                test_length = min(in_len*2, 8192)
                while test_length not in [32, 256, 1024, 2048, 8192]:
                    test_length = test_length * 2
                input_str = open(f"prompt/{test_length}.txt", 'r').read()
                # As different tokenizer has different encodings,
                # slice the input_ids to ensure the prompt length is required length.
                input_ids = tokenizer.encode(input_str, return_tensors="pt")
                input_ids = input_ids[:, :in_len]
                true_str = tokenizer.batch_decode(input_ids)[0]
                input_list = [true_str] * batch_size
                input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
                actual_in_len = input_ids.shape[1]
                result[in_out] = []
                for i in range(num_trials + warm_up):
                    st = time.perf_counter()
                    if streaming:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams, streamer=streamer)
                    else:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    output_ids = output_ids.cpu()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    if not streaming:
                        print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                               actual_in_len, actual_out_len, load_time, model.peak_memory])
                    # torch.xpu.empty_cache() # this may make first token slower
            except RuntimeError:
                traceback.print_exc()
                pass
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    model.to('cpu')
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    del model
    gc.collect()
    return result


def run_transformer_int4_fp16_gpu_win(repo_id,
                                      local_model_hub,
                                      in_out_pairs,
                                      warm_up,
                                      num_trials,
                                      num_beams,
                                      low_bit,
                                      cpu_embedding,
                                      batch_size,
                                      streaming):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer, TextStreamer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half()
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                                     trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half()
        model = model.to('xpu')
    elif repo_id in LLAVA_IDS:
        llava_repo_dir = os.environ.get('LLAVA_REPO_DIR')
        sys.path.append(rf"{llava_repo_dir}")
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                          trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half()
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                     trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half()
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s and {}GB".format(load_time, torch.xpu.memory.memory_reserved()/(1024**3)))

    model = BenchmarkWrapper(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            try:
                in_out_len = in_out.split("-")
                in_len = int(in_out_len[0])
                out_len = int(in_out_len[1])
                # As different tokenizer has different encodings,
                # in_len.txt maybe shorter than we need,
                # use much longer context to make sure input length
                test_length = min(in_len*2, 8192)
                while test_length not in [32, 256, 1024, 2048, 8192]:
                    test_length = test_length * 2
                input_str = open(f"prompt/{test_length}.txt", 'r').read()
                # As different tokenizer has different encodings,
                # slice the input_ids to ensure the prompt length is required length.
                input_ids = tokenizer.encode(input_str, return_tensors="pt")
                input_ids = input_ids[:, :in_len]
                true_str = tokenizer.batch_decode(input_ids)[0]
                input_list = [true_str] * batch_size
                input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
                actual_in_len = input_ids.shape[1]
                result[in_out] = []
                for i in range(num_trials + warm_up):
                    st = time.perf_counter()
                    if streaming:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams, streamer=streamer)
                    else:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    output_ids = output_ids.cpu()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    if not streaming:
                        print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                               actual_in_len, actual_out_len, load_time, model.peak_memory])
                    # torch.xpu.empty_cache() # this may make first token slower
            except RuntimeError:
                traceback.print_exc()
                pass
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    model.to('cpu')
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    del model
    gc.collect()
    return result


def run_transformer_int4_loadlowbit_gpu_win(repo_id,
                                            local_model_hub,
                                            in_out_pairs,
                                            warm_up,
                                            num_trials,
                                            num_beams,
                                            low_bit,
                                            cpu_embedding,
                                            batch_size,
                                            streaming):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer, TextStreamer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load BigDL-LLM optimized low bit model
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.load_low_bit(model_path+'-'+low_bit, optimize_model=True, trust_remote_code=True,
                                       use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path+'-'+low_bit, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.load_low_bit(model_path+'-'+low_bit, optimize_model=True, trust_remote_code=True,
                                                  use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path+'-'+low_bit, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAVA_IDS:
        llava_repo_dir = os.environ.get('LLAVA_REPO_DIR')
        sys.path.append(rf"{llava_repo_dir}")
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        model = AutoModelForCausalLM.load_low_bit(model_path+'-'+low_bit, optimize_model=True, trust_remote_code=True,
                                                  use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path+'-'+low_bit, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.load_low_bit(model_path+'-'+low_bit, optimize_model=True, trust_remote_code=True,
                                                  use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path+'-'+low_bit, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s and {}GB".format(load_time, torch.xpu.memory.memory_reserved()/(1024**3)))

    model = BenchmarkWrapper(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            try:
                in_out_len = in_out.split("-")
                in_len = int(in_out_len[0])
                out_len = int(in_out_len[1])
                # As different tokenizer has different encodings,
                # in_len.txt maybe shorter than we need,
                # use much longer context to make sure input length
                test_length = min(in_len*2, 8192)
                while test_length not in [32, 256, 1024, 2048, 8192]:
                    test_length = test_length * 2
                input_str = open(f"prompt/{test_length}.txt", 'r').read()
                # As different tokenizer has different encodings,
                # slice the input_ids to ensure the prompt length is required length.
                input_ids = tokenizer.encode(input_str, return_tensors="pt")
                input_ids = input_ids[:, :in_len]
                true_str = tokenizer.batch_decode(input_ids)[0]
                input_list = [true_str] * batch_size
                input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu')
                actual_in_len = input_ids.shape[1]
                result[in_out] = []
                for i in range(num_trials + warm_up):
                    st = time.perf_counter()
                    if streaming:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams, streamer=streamer)
                    else:
                        output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                    num_beams=num_beams)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    output_ids = output_ids.cpu()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    if not streaming:
                        print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                               actual_in_len, actual_out_len, load_time, model.peak_memory])
                    # torch.xpu.empty_cache() # this may make first token slower
            except RuntimeError:
                traceback.print_exc()
                pass
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    model.to('cpu')
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    del model
    gc.collect()
    return result


def run_transformer_autocast_bf16( repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in bf16,
    # which convert the relevant layers in the model into BF16 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    model = BenchmarkWrapper(model)

    result = {}
    with torch.inference_mode(), torch.autocast("cpu"):
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
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
                                          actual_in_len, actual_out_len, load_time])
    return result


def run_bigdl_ipex_bf16(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    os.environ["BIGDL_OPT_IPEX"] = "true"

    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in bf16,
    # which convert the relevant layers in the model into BF16 format
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(model.config, "token_latency"):
        model.config.token_latency = True
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    result = {}
    with torch.inference_mode(), torch.autocast("cpu"):
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids, total_list = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                end = time.perf_counter()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                print(output[0])
                actual_out_len = output_ids.shape[1] - actual_in_len
                if i >= warm_up:
                    result[in_out].append([total_list[0], np.mean(total_list[1:]), 0,
                                          actual_in_len, actual_out_len, load_time])
    return result


def run_bigdl_ipex_int4(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    os.environ["BIGDL_OPT_IPEX"] = "true"

    model_path = get_model_path(repo_id, local_model_hub)

    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True, torch_dtype='auto',
                                          use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True, torch_dtype='auto',
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True, torch_dtype='auto',
                                                     use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(model.config, "token_latency"):
        model.config.token_latency = True
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    result = {}
    with torch.inference_mode(), torch.autocast("cpu"):
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids, total_list = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                end = time.perf_counter()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                print(output[0])
                actual_out_len = output_ids.shape[1] - actual_in_len
                if i >= warm_up:
                    result[in_out].append([total_list[0], np.mean(total_list[1:]), 0,
                                          actual_in_len, actual_out_len, load_time])
    return result


def run_bigdl_ipex_int8(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    os.environ["BIGDL_OPT_IPEX"] = "true"

    model_path = get_model_path(repo_id, local_model_hub)

    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True, torch_dtype='auto',
                                          use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True, torch_dtype='auto',
                                                     use_cache=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True, torch_dtype='auto',
                                                     use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(model.config, "token_latency"):
        model.config.token_latency = True
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    result = {}
    with torch.inference_mode(), torch.autocast("cpu"):
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids, total_list = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                end = time.perf_counter()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                print(output[0])
                actual_out_len = output_ids.shape[1] - actual_in_len
                if i >= warm_up:
                    result[in_out].append([total_list[0], np.mean(total_list[1:]), 0,
                                          actual_in_len, actual_out_len, load_time])
    return result


def run_deepspeed_optimize_model_gpu(repo_id,
                                     local_model_hub,
                                     in_out_pairs,
                                     warm_up,
                                     num_trials,
                                     num_beams,
                                     low_bit,
                                     batch_size,
                                     cpu_embedding):
    def get_int_from_env(env_keys, default):
        for e in env_keys:
            val = int(os.environ.get(e, -1))
            if val >= 0:
                return val
        return int(default)
    local_rank = get_int_from_env(["LOCAL_RANK","PMI_RANK"], "0")
    world_size = get_int_from_env(["WORLD_SIZE","PMI_SIZE"], "1")
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    from ipex_llm import optimize_model
    import intel_extension_for_pytorch as ipex
    import deepspeed
    from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
    from deepspeed.accelerator import set_accelerator, get_accelerator
    from intel_extension_for_deepspeed import XPU_Accelerator

    model_path = get_model_path(repo_id, local_model_hub)
    print('model_path:', model_path)
    # First use CPU as accelerator
    # Convert to deepspeed model and apply bigdl-llm optimization on CPU to decrease GPU memory usage
    current_accel = CPU_Accelerator()
    set_accelerator(current_accel)
    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True,
                                          torch_dtype=torch.float16, trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True,
                                                     torch_dtype=torch.float16, trust_remote_code=True, use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True,
                                                     torch_dtype=torch.float16, trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = deepspeed.init_inference(model, mp_size=world_size,
                                     dtype=torch.bfloat16, replace_method="auto",)
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    # Use bigdl-llm `optimize_model` to convert the model into optimized low bit format
    # Convert the rest of the model into float16 to reduce allreduce traffic
    model = optimize_model(model.module.to(f'cpu'), low_bit=low_bit, cpu_embedding=cpu_embedding).to(torch.float16)
    # Next, use XPU as accelerator to speed up inference
    current_accel = XPU_Accelerator()
    set_accelerator(current_accel)
    # Move model back to xpu
    model = model.to(f'xpu:{local_rank}')

    # Modify backend related settings 
    if world_size > 1:
        get_accelerator().set_device(local_rank)
    dist_backend = get_accelerator().communication_backend_name()
    import deepspeed.comm.comm
    deepspeed.comm.comm.cdb = None
    from deepspeed.comm.comm import init_distributed
    init_distributed()

    model = BenchmarkWrapper(model, do_print=True)

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to(f'xpu:{local_rank}')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                torch.xpu.empty_cache()
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                               actual_in_len, actual_out_len, load_time, model.peak_memory])
    del model
    torch.xpu.empty_cache()
    return result


def run_speculative_cpu(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer
    from ipex_llm.transformers.convert import get_enable_ipex

    _enable_ipex = get_enable_ipex()

    model_path = get_model_path(repo_id, local_model_hub)

    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          use_cache=True, speculative=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True, speculative=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='bf16', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     use_cache=True, speculative=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            inputs = tokenizer(input_list, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                if _enable_ipex:
                    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams, attention_mask=attention_mask)
                else:
                    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                end = time.perf_counter()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                print(output[0])
                actual_out_len = output_ids.shape[1] - actual_in_len
                if i >= warm_up:
                    e2e_time = end - st
                    rest_cost_mean = (e2e_time - model.first_token_time)/(model.n_token_generated - 1)
                    result[in_out].append([model.first_token_time, rest_cost_mean, 0,
                                          actual_in_len, actual_out_len, load_time])
    return result


def run_speculative_gpu(repo_id,
                    local_model_hub,
                    in_out_pairs,
                    warm_up,
                    num_trials,
                    num_beams,
                    batch_size):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, LlamaTokenizer

    model_path = get_model_path(repo_id, local_model_hub)

    st = time.perf_counter()
    if repo_id in CHATGLM_IDS:
        model = AutoModel.from_pretrained(model_path, load_in_low_bit='fp16', trust_remote_code=True, torch_dtype=torch.float16,
                                          use_cache=True, speculative=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='fp16', trust_remote_code=True, torch_dtype=torch.float16,
                                                     use_cache=True, speculative=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='fp16', trust_remote_code=True, torch_dtype=torch.float16,
                                                     use_cache=True, speculative=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s".format(load_time))

    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to(model.device)
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                if i >= warm_up:
                    e2e_time = end - st
                    rest_cost_mean = (e2e_time - model.first_token_time)/(model.n_token_generated - 1)
                    result[in_out].append([model.first_token_time, rest_cost_mean, 0,
                                          actual_in_len, actual_out_len, load_time])
    del model
    torch.xpu.empty_cache()
    return result


def run_pipeline_parallel_gpu(repo_id,
                              local_model_hub,
                              in_out_pairs,
                              warm_up,
                              num_trials,
                              num_beams,
                              low_bit,
                              batch_size,
                              cpu_embedding,
                              fp16=False,
                              n_gpu=2):
    from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
    import intel_extension_for_pytorch as ipex
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    origin_repo_id = repo_id.replace("-4bit", "")
    if origin_repo_id in CHATGLM_IDS:
        if "4bit" in repo_id:
            model = AutoModel.load_low_bit(model_path, optimize_model=True,
                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()  
        else:
            model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                            trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cpu_embedding=cpu_embedding)
    elif origin_repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        if "4bit" in repo_id:
            model = AutoModelForCausalLM.load_low_bit(model_path, optimize_model=True,
                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        else:
            if 'starcoder' in repo_id:
                # Load starcoder-15.5b model in bf16 format to avoid CPU OOM.
                model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding, torch_dtype=torch.bfloat16).eval()
                # Convert the low-bit model back to fp32 for performance considerations.
                model = model.float()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                            trust_remote_code=True, use_cache=True, cpu_embedding=cpu_embedding).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if fp16:
        model = model.half()
        print("Convert model to half precision")

    end = time.perf_counter()
    load_time = end - st
    print(">> loading of model costs {}s and {}GB".format(load_time, torch.xpu.memory.memory_reserved()/(1024**3)))

    model_layers = ['model.embed_tokens']
    for i in range(model.config.num_hidden_layers):
        model_layers.append(f'model.layers.{i}')
    model_layers = model_layers + ['model.norm', 'lm_head']

    device_map = {}
    split_len = len(model_layers) // n_gpu
    for i in range(n_gpu):
        device_map.update({key: f'xpu:{i}' for key in model_layers[split_len * i: split_len * (i + 1)]})
        if i == n_gpu - 1:
            device_map.update({key: f'xpu:{i}' for key in model_layers[split_len * (i + 1): ]})
    print(f">> device map: {device_map}")

    from accelerate import dispatch_model
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=None,
        skip_keys=["past_key_value", "past_key_values"],
    )

    model = BenchmarkWrapper(model)
    result = {}
    with torch.inference_mode():
        for in_out in in_out_pairs:
            in_out_len = in_out.split("-")
            in_len = int(in_out_len[0])
            out_len = int(in_out_len[1])
            # As different tokenizer has different encodings,
            # in_len.txt maybe shorter than we need,
            # use much longer context to make sure input length
            test_length = min(in_len*2, 8192)
            while test_length not in [32, 256, 1024, 2048, 8192]:
                test_length = test_length * 2
            input_str = open(f"prompt/{test_length}.txt", 'r').read()
            # As different tokenizer has different encodings,
            # slice the input_ids to ensure the prompt length is required length.
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            input_ids = input_ids[:, :in_len]
            true_str = tokenizer.batch_decode(input_ids)[0]
            input_list = [true_str] * batch_size
            input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu:0')
            actual_in_len = input_ids.shape[1]
            result[in_out] = []
            for i in range(num_trials + warm_up):
                st = time.perf_counter()
                output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                            num_beams=num_beams)
                torch.xpu.synchronize()
                end = time.perf_counter()
                output_ids = output_ids.cpu()
                print("model generate cost: " + str(end - st))
                output = tokenizer.batch_decode(output_ids)
                actual_out_len = output_ids.shape[1] - actual_in_len
                print(output[0])
                torch.xpu.empty_cache()
                if i >= warm_up:
                    result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                           actual_in_len, actual_out_len, load_time, model.peak_memory, fp16])
    del model
    torch.xpu.empty_cache()
    return result


if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')
    today = date.today()
    if 'exclude' in conf:
        excludes = conf['exclude']
    streaming = False
    use_fp16_torch_dtype = False
    n_gpu = 2
    if 'streaming' in conf:
        streaming = conf['streaming']
    if 'use_fp16_torch_dtype' in conf:
        use_fp16_torch_dtype = conf['use_fp16_torch_dtype']
    if 'n_gpu' in conf:
        n_gpu = conf['n_gpu']
    
    import pandas as pd
    for api in conf.test_api:
        global csv_name
        csv_name = f'{current_dir}/{api}-results-{today}.csv'
        for model in conf.repo_id:
            in_out_pairs = conf['in_out_pairs'].copy()
            if excludes:
                for in_out in conf['in_out_pairs']:
                    model_id_input = model + ':' + in_out.split('-')[0]
                    model_id_input_batch_size = model_id_input + ':' + str(conf['batch_size'])
                    if model_id_input in excludes or model_id_input_batch_size in excludes:
                        in_out_pairs.remove(in_out)
            run_model(model, api, in_out_pairs, conf['local_model_hub'], conf['warm_up'], conf['num_trials'], conf['num_beams'],
                      conf['low_bit'], conf['cpu_embedding'], conf['batch_size'], streaming, use_fp16_torch_dtype, n_gpu)
        df = pd.DataFrame(results, columns=['model', '1st token avg latency (ms)', '2+ avg latency (ms/token)', 'encoder time (ms)',
                                            'input/output tokens', 'batch_size', 'actual input/output tokens', 'num_beams', 'low_bit', 'cpu_embedding',
                                            'model loading time (s)', 'peak mem (GB)', 'streaming', 'use_fp16_torch_dtype'])
        df.to_csv(csv_name)
        results = []
