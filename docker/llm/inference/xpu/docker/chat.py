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
SYSTEM_PROMPT = "A chat between a curious human <human> and an artificial intelligence assistant <bot>.\
The assistant gives helpful, detailed, and polite answers to the human's questions."
HUMAN_ID = "<human>"
BOT_ID = "<bot>"
# chat_history formated in [(iput_str, output_str)]
def format_prompt(input_str,
                  chat_history):
    prompt = [f"{SYSTEM_PROMPT}\n"]
    for history_input_str, history_output_str in chat_history:
        prompt.append(f"{HUMAN_ID} {history_input_str}\n{BOT_ID} {history_output_str}\n")
    prompt.append(f"{HUMAN_ID} {input_str}\n{BOT_ID} ")
    return "".join(prompt)
def stream_chat(model,
                tokenizer,
                stopping_criteria,
                input_str,
                chat_history):
    prompt = format_prompt(input_str, chat_history)
    # print(prompt)
    input_ids = tokenizer([prompt], return_tensors="pt").to('xpu')
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(input_ids, streamer=streamer, max_new_tokens=512, stopping_criteria=stopping_criteria)
    from threading import Thread
    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    output_str = []
    print(Fore.BLUE+"BigDL-LLM: "+Fore.RESET, end="")
    for partial_output_str in streamer:
        output_str.append(partial_output_str)
        # remove the last HUMAN_ID if exists
        print(partial_output_str.replace(f"{HUMAN_ID}", ""), end="")
    chat_history.append((input_str, "".join(output_str).replace(f"{HUMAN_ID}", "").rstrip()))
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
  model = model.to('xpu')
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(HUMAN_ID, tokenizer)])
  chat_history = []
  while True:
      with torch.inference_mode():
          user_input = input(Fore.GREEN+"\nHuman: "+Fore.RESET)
          if user_input == "stop": # let's stop the conversation when user input "stop"
              break
          stream_chat(model=model,
                      tokenizer=tokenizer,
                      stopping_criteria=stopping_criteria,
                      input_str=user_input,
                      chat_history=chat_history)
                      
