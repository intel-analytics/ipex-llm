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

from bigdl.llm import optimize_model
from kv_cache import StartRecentKVCache

HUMAN_ID = "<human>"
BOT_ID = "<bot>"

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    print(Fore.BLUE+"BigDL-LLM: "+Fore.RESET, end="")
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
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
    print(" ".join(generated_text[pos:]).strip('\n<'), flush=True)
    return past_key_values

@torch.no_grad()
def stream_chat(model, tokenizer, kv_cache=None, max_gen_len=512):
    past_key_values = None
    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        if user_input == "stop": # let's stop the conversation when user input "stop"
            break
        prompt = f"{HUMAN_ID} {user_input}\n{BOT_ID} "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )

@torch.no_grad()
def chatglm2_stream_chat(model, tokenizer):
    chat_history = []
    past_key_values = None
    current_length = 0
    stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(HUMAN_ID, tokenizer)])
    max_past_length = 2048

    while True:
        user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
        if user_input == "stop": # let's stop the conversation when user input "stop"
            break
        print(Fore.BLUE+"BigDL-LLM: "+Fore.RESET, end="")
        prompt = f"问：{user_input}\n答："
        for response, chat_history, past_key_values in model.stream_chat(tokenizer, prompt,
                                                                         history=chat_history,
                                                                         stopping_criteria=stopping_criteria,
                                                                         past_key_values=past_key_values,
                                                                         return_past_key_values=True):
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
            if past_key_values[0][0].shape[0] > max_past_length:
                # To avoid out of memory, only keep recent key_values
                new_values_list = []
                for i in range(len(past_key_values)):
                    new_value = []
                    for val in past_key_values[i]:
                        new_v = val[-max_past_length:]
                        new_value.append(new_v)
                    new_values_list.append(tuple(new_value))
                past_key_values = tuple(new_values_list)

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
    args = parser.parse_args()

    model_path = args.model_path

    model = auto_select_model(model_path)
    model = optimize_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if model.config.architectures is not None and model.config.architectures[0] == "ChatGLMModel":
        chatglm2_stream_chat(model=model, tokenizer=tokenizer)
    else:
        kv_cache = StartRecentKVCache()
        stream_chat(model=model,
                    tokenizer=tokenizer,
                    kv_cache=kv_cache)
