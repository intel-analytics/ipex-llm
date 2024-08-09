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
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from ipex_llm.llamaindex.llms import IpexLLM
import argparse

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def main(args):
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    llm = IpexLLM(
        model_name=args.model_path,
        tokenizer_name=args.model_path,
        context_window=512,
        max_new_tokens=args.n_predict,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        device_map="xpu"
    )
    agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)
    response = agent.chat(args.question)
    print("=========response=========")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LlamaIndex IpexLLM Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is 20+(2*4)?',
                        help='qustion you want to ask.')
    parser.add_argument('-n','--n-predict', type=int, default=64,
                        help='max number of predict tokens')
    args = parser.parse_args()
    
    main(args)