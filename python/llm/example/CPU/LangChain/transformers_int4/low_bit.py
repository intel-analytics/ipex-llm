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


import argparse

from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFacePipeline


def main(args):
    question = args.question
    model_path = args.model_path
    low_bit_model_path = args.target_path
    template ="""{question}"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = TransformersLLM.from_model_id(
        model_id=model_path,
        model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
    )
    llm.model.save_low_bit(low_bit_model_path)
    del llm
    low_bit_llm = TransformersLLM.from_model_id_low_bit(
        model_id=low_bit_model_path,
        tokenizer_id=model_path,
        model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True}
    )
    llm_chain = LLMChain(prompt=prompt, llm=low_bit_llm)

    output = llm_chain.run(question)
    print("====output=====")
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain Chat Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-t','--target-path',type=str,required=True,
                        help='the path to save the low bit model')
    parser.add_argument('-q', '--question', type=str, default='What is AI?',
                        help='qustion you want to ask.')
    args = parser.parse_args()
    
    main(args)