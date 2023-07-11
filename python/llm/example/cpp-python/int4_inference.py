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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

import argparse

def main(args):
    model_family = args.model_family
    model_path = args.model_path
    prompt = args.prompt
    n_threads = args.thread_num
    
    if model_family == "llama":     
        from bigdl.llm.models import Llama
        modelclass = Llama
    if model_family == "bloom":   
        from bigdl.llm.models import Bloom
        modelclass = Bloom
    if model_family == "gptneox": 
        from bigdl.llm.models import Gptneox  
        modelclass = Gptneox
    if model_family == "starcoder":   
        from bigdl.llm.models import Starcoder
        modelclass = Starcoder
        
    model = modelclass(model_path, n_threads=n_threads)
    response=model(prompt)
    print(response)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Llama-CPP-Python style API Simple Example')
    parser.add_argument('-x','--model-family', type=str, required=True,
                        choices=["llama", "bloom", "gptneox", "starcoder"],
                        help='the model family')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to the converted llm model')
    parser.add_argument('-p', '--prompt', type=str, default='What is AI?',
                        help='qustion you want to ask.')
    parser.add_argument('-t','--thread-num', type=int, default=2,
                        help='number of threads to use for inference')
    args = parser.parse_args()
    
    main(args)
