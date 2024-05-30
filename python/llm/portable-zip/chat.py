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
# https://github.com/mit-han-lab/streaming-llm/blob/main/examples/run_streaming_llama.py
# which is licensed under the MIT license:
#
# MIT License
# 
# Copyright (c) 2023 MIT HAN Lab
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import argparse
import sys

# todo: support more model class
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer
from transformers.tools.agents import StopSequenceCriteria
from transformers.generation.stopping_criteria import StoppingCriteriaList

from colorama import Fore

from ipex_llm import optimize_model
from kv_cache import StartRecentKVCache

HUMAN_ID = "<human>"
BOT_ID = "<bot>"

def get_stop_words_ids(chat_format, tokenizer):
    # https://github.com/QwenLM/Qwen/blob/main/examples/vllm_wrapper.py#L23
    if chat_format == "Qwen":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id], [tokenizer.eod_id]]
    # https://huggingface.co/01-ai/Yi-6B-Chat/blob/main/tokenizer_config.json#L38
    elif chat_format == "Yi":
        stop_words_ids = [tokenizer.encode("<|im_end|>")]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, stop_words=[]):
    print(Fore.BLUE+"IPEX-LLM: "+Fore.RESET, end="")
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    stop = False
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if stop_words is not None:
            for stop_str in stop_words:
                if generated_ids[-1 * len(stop_str):] == stop_str:
                    stop = True
                    break
            if stop:
                break
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True,
                                          spaces_between_special_tokens=False)

        now = len(generated_text) - 1
        if now > pos:
            if '\n<' in generated_text:
                break
            else:
                print("".join(generated_text[pos:now]), end="", flush=True)
                pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    if generated_text:
        print(" ".join(generated_text[pos:]).strip('\n<'), flush=True)
    return past_key_values

@torch.no_grad()
def stream_chat(model, tokenizer, kv_cache=None, max_gen_len=512, stop_words=[]):
    past_key_values = None
    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        # let's stop the conversation when user input "stop"
        if user_input == "stop":
            break
        prompt = f"{HUMAN_ID} {user_input}\n{BOT_ID} "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, stop_words=stop_words
        )

@torch.no_grad()
def chatglm3_stream_chat(model, tokenizer):
    chat_history = []
    past_key_values = None
    current_length = 0
    # https://github.com/THUDM/ChatGLM3/issues/274#issuecomment-1810160305
    stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(["<|user|>", "<|observation|>"], tokenizer)])
    # you could change this according to your memory requirement
    max_past_length = 512
    block_length = 512

    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        # let's stop the conversation when user input "stop"
        if user_input == "stop":
            break
        print(Fore.BLUE+"IPEX-LLM: "+Fore.RESET, end="")
        # https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md
        prompt = f"""
            <|system|>
            You are an intelligent AI assistant, named ChatGLM3. Follow the user's instructions carefully.
            <|user|>
            {user_input}
            <|assistant|>
        """
        if past_key_values is not None and past_key_values[0][0].shape[0] > max_past_length + block_length:
            # To avoid out of memory, only keep recent key_values of max_past_length
            past_key_values = [(k[-max_past_length:, :, :, :], v[-max_past_length:, :, :, :]) for k, v in past_key_values]
        for response, chat_history, past_key_values in model.stream_chat(tokenizer, prompt,
                                                                         history=chat_history,
                                                                         stopping_criteria=stopping_criteria,
                                                                         past_key_values=past_key_values,
                                                                         return_past_key_values=True):
            print(response[current_length:], end="", flush=True)
            current_length = len(response)

@torch.no_grad()
def qwen_stream_chat(model, tokenizer, kv_cache=None, max_gen_len=512, stop_words=[]):
    past_key_values = None
    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        # let's stop the conversation when user input "stop"
        if user_input == "stop":
            break
        # https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/generation_config.json#L2
        prompt = f"""
            <|im_start|>system
            You are a helpful assistant.
            <|im_end|>
            <|im_start|>user
            {user_input}
            <|im_end|>
            <|im_start|>assistant
        """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, stop_words=stop_words
        )

@torch.no_grad()
def llama_stream_chat(model, tokenizer, kv_cache=None, max_gen_len=512, stop_words=[]):
    past_key_values = None
    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        # let's stop the conversation when user input "stop"
        if user_input == "stop":
            break
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML#prompt-template-llama-2-chat
        prompt = f"""
            [INST] <<SYS>>
            You are a helpful assistant.
            <</SYS>>
            {user_input}[/INST]
        """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, stop_words=stop_words
        )

@torch.no_grad()
def yi_stream_chat(model, tokenizer, kv_cache=None, max_gen_len=512, stop_words=[]):
    past_key_values = None
    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        # let's stop the conversation when user input "stop"
        if user_input == "stop":
            break
        # https://huggingface.co/01-ai/Yi-6B-Chat#31-use-the-chat-model
        prompt = f"""
            <|im_start|>system
            You are a helpful assistant. If you don't understand what the user means, ask the user to provide more information.
            <|im_end|>
            <|im_start|>user
            {user_input}
            <|im_end|>
            <|im_start|>assistant
        """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, stop_words=stop_words
        )


def format_prompt_with_history(input_str,
                  chat_history):
    SYSTEM_PROMPT = "A chat between a curious human <human> and an artificial intelligence assistant <bot>.\
    The assistant gives helpful, detailed, and polite answers to the human's questions."
    prompt = [f"{SYSTEM_PROMPT}\n"]
    # prompt = []
    for history_input_str, history_output_str in chat_history:
        prompt.append(f"{HUMAN_ID} {history_input_str}\n{BOT_ID} {history_output_str}\n")
    prompt.append(f"{HUMAN_ID} {input_str}\n{BOT_ID} ")

    return "".join(prompt)


def stream_chat_with_history(model, tokenizer):
    stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(HUMAN_ID, tokenizer)])

    chat_history = []

    while True:
        with torch.inference_mode():
            user_input = input(Fore.GREEN + "\nHuman: " + Fore.RESET)
            if user_input == "stop":  # let's stop the conversation when user input "stop"
                break
            prompt = format_prompt_with_history(user_input, chat_history)
            # print(prompt)
            input_ids = tokenizer([prompt], return_tensors="pt")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(input_ids, streamer=streamer, max_new_tokens=512,
                                   stopping_criteria=stopping_criteria)

            from threading import Thread
            # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()

            output_str = []
            print(Fore.BLUE + "IPEX-LLM: " + Fore.RESET, end="")
            for partial_output_str in streamer:
                output_str.append(partial_output_str)
                # remove the last HUMAN_ID if exists
                print(partial_output_str.replace(f"{HUMAN_ID}", ""), end="")

            chat_history.append((user_input, "".join(output_str).replace(f"{HUMAN_ID}", "").rstrip()))

def auto_select_model(model_name):
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype="auto",
                                                        trust_remote_code=True,
                                                        use_cache=True)
        except:
            model = AutoModel.from_pretrained(model_path,
                                             low_cpu_mem_usage=True,
                                             torch_dtype="auto",
                                             trust_remote_code=True,
                                             use_cache=True)
    except:
        print("Sorry, the model you entered is not supported in installer.")
        sys.exit()
    
    return model
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="path to an llm")
    parser.add_argument("--start-size", type=int, default=4, help="start_size of kv_cahce")
    parser.add_argument("--recent-size", type=int, default=2000)
    args = parser.parse_args()

    model_path = args.model_path
    start_size = args.start_size
    recent_size = args.recent_size

    model = auto_select_model(model_path)
    model = optimize_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if model.config.architectures is not None and model.config.architectures[0] == "QWenLMHeadModel":
        stop_words = get_stop_words_ids("Qwen", tokenizer=tokenizer)
        kv_cache = StartRecentKVCache(start_size=start_size,
                                      k_seq_dim=1,
                                      v_seq_dim=1,
                                      recent_size=recent_size)
        qwen_stream_chat(model=model, tokenizer=tokenizer,kv_cache=kv_cache, stop_words=stop_words)
    elif model.config.architectures is not None and model.config.architectures[0] == "ChatGLMModel":
        chatglm3_stream_chat(model=model, tokenizer=tokenizer)
    elif model.config.architectures is not None and model.config.architectures[0] == "LlamaForCausalLM":
        kv_cache = StartRecentKVCache(start_size=start_size, recent_size=recent_size)
        if "yi" in model_path.lower():
            stop_words = get_stop_words_ids("Yi", tokenizer=tokenizer)
            yi_stream_chat(model=model, tokenizer=tokenizer, kv_cache=kv_cache, stop_words=stop_words)
        else:
            llama_stream_chat(model=model, tokenizer=tokenizer, kv_cache=kv_cache)
    elif model.config.architectures[0] == "BaichuanForCausalLM" and model.config.vocab_size == 64000:
        # Baichuan-13B-Chat
        stream_chat_with_history(model=model, tokenizer=tokenizer)
    else:
        kv_cache = StartRecentKVCache(start_size=start_size, recent_size=recent_size)
        stream_chat(model=model,
                    tokenizer=tokenizer,
                    kv_cache=kv_cache)
