
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

# Code is adapted from https://python.langchain.com/docs/modules/chains/additional/llm_math

import argparse
import warnings

from langchain.chains import LLMMathChain
from langchain_community.llms import IpexLLM

warnings.filterwarnings("ignore", category=UserWarning, message=".*padding_mask.*")


def main(args):
    
    question = args.question
    model_path = args.model_path
    
    llm = IpexLLM.from_model_id(
        model_id=model_path,
        model_kwargs={
            "temperature": 0,
            "max_length": 1024,
            "trust_remote_code": True,
        },
    )
    
    llm_math = LLMMathChain.from_llm(llm, verbose=True)

    output = llm_math.run(question)
    print("====output=====")
    print(output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain Math Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is 13 raised to the .3432 power?',
                        help='qustion you want to ask.')
    args = parser.parse_args()
    
    main(args)