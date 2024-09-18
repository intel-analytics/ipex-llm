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
# This file is adapted from
# https://github.com/THUDM/LongBench/blob/main/pred.py
# and
# https://github.com/FasterDecoding/SnapKV/blob/main/experiments/LongBench/pred_snap.py


import os
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))

valid_model_names = [
    "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
    "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
    "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m",
    "qwen2-7b-instruct", "chatglm4-9b"]

valid_datasets_e = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

valid_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"] + \
                ["multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"]

valid_dtypes = ['fp16', 'fp32']


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        print('chatglm3')
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm2" in model_name:
        print('chatglm2')
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        print('longchat')
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2"  in model_name or "llama-2" in model_name or "lwm" in model_name:
        print('llama2', model_name)
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        print('xgen')
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        print('internlm')
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "mistral" in model_name or "mixtral" in model_name:
        print('mistral')
        prompt = prompt
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, 
                        prompt_format, dataset, model_name, 
                        model2path, out_path, low_bit, dtype, optimize_model,
                        compress=False, 
                        window_sizes = None,
                        default_max_capacity_prompts = None,
                        specific_max_capcity_prompts = None,
                        kernel_sizes = None,
                        pooling = None):

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device = "xpu", dtype_=dtype, low_bit=low_bit, optimize_model=optimize_model)
    device = model.device
    print(f"model_device: {model.device}")
    printed = False
    print(out_path)
    count_prompt_under_maxlen = 0
    for json_obj in tqdm(data):
        ############################################################################################################
        # load compress args
        count_prompt_under_maxlen += 1
        if compress:
            inner_model = model.model if hasattr(model, "model") else model.base_model.encoder
            layers = len(inner_model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            max_capacity_prompts = [default_max_capacity_prompts] * layers
            if specific_max_capcity_prompts is not None:
                for key, value in specific_max_capcity_prompts.items():
                    max_capacity_prompts[key] = value
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            from transformers.configuration_utils import PretrainedConfig
            for i in range(layers):
                cur_layer = inner_model.layers[i]
                cur_layer_attn = cur_layer.self_attn if hasattr(cur_layer, "self_attn") else cur_layer.self_attention
                cur_layer_attn.config = cur_layer_attn.config if hasattr(cur_layer_attn, "config") else PretrainedConfig()

                cur_layer_attn.config.window_size = window_sizes[i]
                cur_layer_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                cur_layer_attn.config.kernel_size = kernel_sizes[i]
                cur_layer_attn.config.pooling = pooling
        ############################################################################################################
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        #print(f'initial len = {tokenized_prompt.shape}')
        if len(tokenized_prompt) > max_length:
            count_prompt_under_maxlen -= 1
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        print(f'context_length = {context_length}')
        if not printed:
            print(prompt)
            printed = True
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    count_out_path = os.path.join(os.path.split(out_path)[0], "uncut_prompt_count.json")
    prompt_count_result = {}
    if os.path.isfile(count_out_path):
        with open(count_out_path, "r", encoding = "utf-8") as f:
            prompt_count_result = json.load(f)
    prompt_count_result[dataset] = count_prompt_under_maxlen
    with open(count_out_path, "w", encoding = "utf-8") as f:
        json.dump(prompt_count_result, f, ensure_ascii=False, indent=4)



def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_model_and_tokenizer(path, model_name, device, dtype_, low_bit, optimize_model):
    if (dtype_ == 'fp32'):
        dtype = torch.float32
    elif (dtype_ == 'fp16'):
        dtype = torch.float16
    else:
        raise ValueError(f"dtype {dtype_} is not supported")
    model = AutoModelForCausalLM.from_pretrained(
                path,
                optimize_model=optimize_model,
                load_in_low_bit=low_bit,
                use_cache=True,
                trust_remote_code=True,
                torch_dtype = dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
    )
    model = model.half().to(device)
    return model, tokenizer

def compresskv_config_range(full_kv: bool, configs: list[str], model_name: str):
    if full_kv:
        os.environ["IPEX_LLM_COMPRESS_KV_CACHE"] = "0"
        yield False, {}, model_name

    os.environ["IPEX_LLM_COMPRESS_KV_CACHE"] = "1"
    for config in configs:
        yield True, json.load(open(os.path.join(f'{current_dir}/config', f"{config}.json"), "r")), f"{model_name}_{config}"


if __name__ == '__main__':
    seed_everything(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    model_names = conf['model_name'] if OmegaConf.is_list(conf['model_name']) else [conf['model_name']]
    full_kv = conf['full_kv']
    e = conf['e']
    compresskv_configs = conf['compress_kv'] if OmegaConf.is_list(conf['compress_kv']) else [conf['compress_kv']]
    datasets = conf['datasets'] if OmegaConf.is_list(conf['datasets']) else [conf['datasets']]
    dtype = conf['dtype']
    low_bit = conf['low_bit']
    optimize_model = conf['optimize_model']

    model2path = json.load(open(f"{current_dir}/config/model2path.json", "r"))
    model2maxlen = json.load(open(f"{current_dir}/config/model2maxlen.json", "r"))

    dataset2prompt = json.load(open(f"{current_dir}/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{current_dir}/config/dataset2maxlen.json", "r"))

    ## check
    for model_name in model_names:
        if model_name not in valid_model_names:
            raise ValueError(f"model {model_name} is not supported")
    if e not in [True, False]:
        raise ValueError("e should be True or False")
    for dataset in datasets:
        if e:
            valid_dataset_check = valid_datasets_e
        else:
            valid_dataset_check = valid_datasets
        # check if args dataset in datasets
        if dataset not in valid_dataset_check:
            raise ValueError(f"Dataset {dataset} not found in datasets")
    if dtype not in valid_dtypes:
        raise ValueError(f"dtype {dtype} is not supported")
    
    for model_name in model_names:
        max_length = model2maxlen[model_name]
        for compress, compress_args, write_model_name in compresskv_config_range(full_kv, compresskv_configs, model_name):
            for dataset in datasets:
                e_string = "_e" if e else ""
                data = load_dataset('THUDM/LongBench', f"{dataset}{e_string}", split='test')

                if not os.path.exists(f"{current_dir}/pred{e_string}_{max_length}"):
                    os.makedirs(f"{current_dir}/pred{e_string}_{max_length}")
                if not os.path.exists(f"{current_dir}/pred{e_string}_{max_length}/{write_model_name}"):
                    os.makedirs(f"{current_dir}/pred{e_string}_{max_length}/{write_model_name}")
                out_path = f"{current_dir}/pred{e_string}_{max_length}/{write_model_name}/{dataset}.jsonl"
                
                prompt_format = dataset2prompt[dataset]
                max_gen = dataset2maxlen[dataset]
                data_all = [data_sample for data_sample in data]
                get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, low_bit, dtype, compress, optimize_model, **compress_args)
