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
benchmark_util_path = os.path.join(current_dir, '..')
import sys
sys.path.append(benchmark_util_path)
from benchmark_util import BenchmarkWrapper
from bigdl.llm.utils.common.log4Error import invalidInputError

import subprocess
import matplotlib.pyplot as plt

LLAMA_IDS = ['meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-13b-chat-hf',
             'meta-llama/Llama-2-70b-chat-hf','decapoda-research/llama-7b-hf',
             'decapoda-research/llama-65b-hf','lmsys/vicuna-7b-v1.5',
             'lmsys/vicuna-13b-v1.3','project-baize/merged-baize-30b']

CHATGLM_IDS = ['THUDM/chatglm-6b', 'THUDM/chatglm2-6b', 'THUDM/chatglm3-6b']

LLAVA_IDS = ['liuhaotian/llava-v1.5-7b']

results = []
excludes = []

# To record and plot the memory usage
class MonitorMemoryUsage:
    def __init__(self, index):
        self.x = []
        self.data = []
        self.monitoring = False
        self.index = index

    def start_monitoring(self):
        self.monitoring = True
        self.start_time = time.time()
        self.proc = subprocess.Popen(['top', '-d', '0.01', '-b'], stdout=subprocess.PIPE, text=True)

        # Start a new thread to monitor memory usage
        self.thread = threading.Thread(target=self.capture_output)
        self.thread.start()

    def capture_output(self):
        while self.monitoring:
            line = self.proc.stdout.readline()
            if "MiB Mem" in line:
                # Extract memory usage data
                timestamp = time.time() - self.start_time
                # Adjust the index based on your output
                mem_usage = float(line.split()[self.index])  
                self.x.append(timestamp * 100)
                self.data.append(mem_usage)

    def stop_monitoring(self, output_dir=".", name=""):
        self.monitoring = False
        self.proc.terminate()
        self.thread.join()
        # Plot the memory usage
        plt.plot(self.x, self.data, marker='o', linestyle='-')
        plt.xlabel('Time/10ms')
        plt.ylabel('Used/MB')
        plt.title('Used Memory Over Time')
        # plt.legend()
        plt.grid(True)
        plt.ylim(min(self.data)-100, max(self.data)+200)
        plt.savefig(f'{output_dir}/memory_usage_plot_load_{name}.png')
        plt.close()


def run_model_in_thread(model, in_out, tokenizer, result, warm_up, num_beams, input_ids, out_len, actual_in_len, num_trials):
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
                                   actual_in_len, actual_out_len, model.peak_memory])

def run_model(repo_id, test_api, in_out_pairs, local_model_hub=None, warm_up=1, num_trials=3, num_beams=1, low_bit='sym_int4', cpu_embedding=False, batch_size=1 ,index=7, plot_memory_vs_time_graph=False):
    # TODO: make a parameter
    result= {}
    
    if not os.path.exists("./output/"):
        os.makedirs("./output")

    if plot_memory_vs_time_graph == True:
        monitor = MonitorMemoryUsage(index)
        monitor.start_monitoring()

    if test_api == 'transformer_int4':
        result = run_transformer_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'native_int4':
        run_native_int4(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials)
    elif test_api == 'optimize_model':
        result = run_optimize_model(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
    elif test_api == 'transformer_int4_gpu':
        result = run_transformer_int4_gpu(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, batch_size)
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
        result = run_transformer_int4_gpu_win(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, low_bit, cpu_embedding, batch_size)
    elif test_api == 'transformer_autocast_bf16':
        result = run_transformer_autocast_bf16(repo_id, local_model_hub, in_out_pairs, warm_up, num_trials, num_beams, batch_size)

    if plot_memory_vs_time_graph == True:
        monitor.stop_monitoring("./output", test_api + "_" + in_out_pairs[0])

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
                            result[in_out_pair][-1][5] if 'int4_gpu' in test_api else 'N/A']) # currently only peak mem for transformer_int4_gpu is caught here


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
                         num_trials,
                         num_beams,
                         low_bit,
                         batch_size):
    from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
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
                                           actual_in_len, actual_out_len])
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
    print(">> loading of model costs {}s".format(end - st))

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
                                           actual_in_len, actual_out_len])
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
    from bigdl.llm import optimize_model

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
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True, trust_remote_code=True).eval()
        model = optimize_model(model, low_bit=low_bit)
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
                                           actual_in_len, actual_out_len])
    return result


def run_transformer_int4_gpu(repo_id,
                             local_model_hub,
                             in_out_pairs,
                             warm_up,
                             num_trials,
                             num_beams,
                             low_bit,
                             batch_size):
    from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
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
                                            trust_remote_code=True, use_cache=True).eval()  
        else:
            model = AutoModel.from_pretrained(model_path, load_in_low_bit=low_bit, optimize_model=True,
                                            trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    elif origin_repo_id in LLAMA_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True).eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
    else:
        if 'starcoder' in repo_id:
            # Load starcoder-15.5b model in bf16 format to avoid CPU OOM. 
            model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                        trust_remote_code=True, use_cache=True, torch_dtype=torch.bfloat16).eval()
            # Convert the low-bit model back to fp32 for performance considerations.
            model = model.float()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=True, load_in_low_bit=low_bit,
                                                        trust_remote_code=True, use_cache=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to('xpu')
        if isinstance(model, GPTJForCausalLM):
            # For gpt-j model family, this optimization can provide a better performance.
            model = ipex.optimize(model.eval(), inplace=True)
    end = time.perf_counter()
    print(">> loading of model costs {}s and {}GB".format(end - st, torch.xpu.memory.memory_reserved()/(1024**3)))

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
            thread = threading.Thread(target=run_model_in_thread, args=(model, in_out, tokenizer, result, warm_up, num_beams, input_ids, out_len, actual_in_len, num_trials))
            thread.start()
            thread.join()
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
    from bigdl.llm import optimize_model
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
        if isinstance(model, GPTJForCausalLM):
            # For gpt-j model family, this optimization can provide a better performance.
            model = ipex.optimize(model.eval(), inplace=True)
    end = time.perf_counter()
    print(">> loading of model costs {}s".format(end - st))

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
                                           actual_in_len, actual_out_len])
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
        if isinstance(model, GPTJForCausalLM):
            # For gpt-j model family, this optimization can provide a better performance.
            model = ipex.optimize(model.eval(), inplace=True)
    end = time.perf_counter()
    print(">> loading of model costs {}s".format(end - st))

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
                                           actual_in_len, actual_out_len])
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
    from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
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
    print(">> loading of model costs {}s".format(end - st))

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
                                           actual_in_len, actual_out_len])
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
    from bigdl.llm import optimize_model
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
    print(">> loading of model costs {}s".format(end - st))

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
                                           actual_in_len, actual_out_len])
    return result


def run_transformer_int4_gpu_win(repo_id,
                                 local_model_hub,
                                 in_out_pairs,
                                 warm_up,
                                 num_trials,
                                 num_beams,
                                 low_bit,
                                 cpu_embedding,
                                 batch_size):
    from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer
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
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True,
                                                     use_cache=True, cpu_embedding=cpu_embedding).eval()
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
        if isinstance(model, GPTJForCausalLM):
            # For gpt-j model family, this optimization can provide a better performance.
            model = ipex.optimize(model.eval(), inplace=True)
    end = time.perf_counter()
    print(">> loading of model costs {}s and {}GB".format(end - st, torch.xpu.memory.memory_reserved()/(1024**3)))

    model = BenchmarkWrapper(model)

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
                    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=out_len,
                                                num_beams=num_beams)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    output_ids = output_ids.cpu()
                    print("model generate cost: " + str(end - st))
                    output = tokenizer.batch_decode(output_ids)
                    print(output[0])
                    actual_out_len = output_ids.shape[1] - actual_in_len
                    if i >= warm_up:
                        result[in_out].append([model.first_cost, model.rest_cost_mean, model.encoder_time,
                                               actual_in_len, actual_out_len, model.peak_memory])
                    # torch.xpu.empty_cache() # this may make first token slower
            except RuntimeError:
                traceback.print_exc()
                pass
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
    from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
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
    print(">> loading of model costs {}s".format(end - st))

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
                                          actual_in_len, actual_out_len])
    return result

if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')
    today = date.today()
    index = 7  # This is the default index, which can be modified by keyboard input
    if 'exclude' in conf:
        excludes = conf['exclude']
    if conf['plot_memory_vs_time_graph'] == True:
        process = subprocess.Popen(['top', '-b', '-n', '1'], stdout=subprocess.PIPE, text=True)
        flag = True
        for line in process.stdout:
            if "MiB Mem" in line:
                mem_line = line
                break
            
        # Instructions to check if the memory usage is correct
        print('''
            The first line below is the original output of the top command, and the second line extracts the memory usage using 'split()[7]' from the original output. 
            However, due to different Linux distributions and versions of the top tool, the output format may vary.
            Therefore, we need to manually check if the memory usage in the second line correctly corresponds to the right location in the first line.
            If it does not, we need to manually adjust the 'split()[]' index to make them match. 
            If it does match, enter 'y', and the program will continue to run; otherwise, enter 'n', then manually modify the index and run it again; enter 'quit' to quit.
            ''')
        while(flag):
            mem_usage = line.split()[index]
            print("-" * 100)
            print("This is the entire output of the top command:\n", mem_line)
            print("This is the extracted memory usage:\n", mem_usage)
            print("-" * 100)
            user_input = input("Please check if the memory usage is correct([y]/n/exit): ").strip().lower()
            if user_input == "y" or user_input == "":
                flag = False
                print("Continuing with the process...")
            elif user_input == "n":
                try:
                    index_input = input("Please type in an integer to update the index:")
                    index = int(index_input)
                except ValueError:
                    print("Please type in an integer!")
            else:
                exit()

    import pandas as pd
    for api in conf.test_api:
        for model in conf.repo_id:
            in_out_pairs_ = conf['in_out_pairs'].copy()
            for in_out_pairs in in_out_pairs_:
                in_out_pairs = [in_out_pairs]
                if excludes:
                    for in_out in conf['in_out_pairs']:
                        model_id_input = model + ':' + in_out.split('-')[0]
                        if model_id_input in excludes:
                            in_out_pairs.remove(in_out)
                run_model(model, api, in_out_pairs, conf['local_model_hub'], conf['warm_up'], conf['num_trials'], conf['num_beams'],
                          conf['low_bit'], conf['cpu_embedding'], conf['batch_size'], index=index, plot_memory_vs_time_graph=conf['plot_memory_vs_time_graph'])

                    
        df = pd.DataFrame(results, columns=['model', '1st token avg latency (ms)', '2+ avg latency (ms/token)', 'encoder time (ms)',
                                            'input/output tokens', 'actual input/output tokens', 'num_beams', 'low_bit', 'cpu_embedding', 
                                            'peak mem (GB)'])

        df.to_csv(f'{current_dir}/output/{api}-results-{today}.csv')
        results = []
