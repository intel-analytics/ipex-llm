import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch
from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m"])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        print('chatglm3')
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        print('chatglm')
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
        # from fastchat.model import get_conversation_template
        # conv = get_conversation_template("mistral")
        # conv.append_message(conv.roles[0], prompt)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
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
                        model2path, out_path, 
                        compress=False, 
                        window_sizes = None,
                        max_capacity_prompts = None,
                        kernel_sizes = None,
                        pooling = None):
    # device = torch.device(f'cuda:{rank}')
    # device = model.device
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device = "cuda", compress=compress)
    device = model.device
    printed = False
    print(out_path)
    for json_obj in tqdm(data):
        ############################################################################################################
        # load compress args
        if compress:
            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
        ############################################################################################################
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress=False):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
        )
    elif "llama-2" in model_name or "lwm" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
        )
    elif "mistral" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
        )
    elif "mixtral" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            # padding_side="right",
            # use_fast=False,
        )
    else:
        raise ValueError(f"Model {model_name} not supported!")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("/home/arda/yina/SnapKV/experiments/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("/home/arda/yina/SnapKV/experiments/LongBench/config/model2maxlen.json", "r"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    # check if args dataset in datasets
    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not found in datasets")
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("/home/arda/yina/SnapKV/experiments/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/home/arda/yina/SnapKV/experiments/LongBench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists(f"pred_{max_length}"):
        os.makedirs(f"pred_{max_length}")
    if not os.path.exists(f"pred_e_{max_length}"):
        os.makedirs(f"pred_e_{max_length}")
    dataset = args.dataset
    # for dataset in datasets:
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('/home/arda/yina/SnapKV/experiments/LongBench/config', args.compress_args_path), "r"))
        compress = True
        write_model_name = model_name + args.compress_args_path.split(".")[0]
        replace_llama()
        replace_mistral()
        replace_mixtral()
    else:
        compress = False
        compress_args = None
        write_model_name = model_name
    if args.e:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        if not os.path.exists(f"pred_e_{max_length}/{write_model_name}"):
            os.makedirs(f"pred_e_{max_length}/{write_model_name}")
        out_path = f"pred_e_{max_length}/{write_model_name}/{dataset}.jsonl"
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        if not os.path.exists(f"pred_e_{max_length}/{write_model_name}"):
            os.makedirs(f"pred_e_{max_length}/{write_model_name}")
        out_path = f"pred_e_{max_length}/{write_model_name}/{dataset}.jsonl"
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]
    if compress_args is not None:
        get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, compress, **compress_args)
    else:
        get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, compress)
