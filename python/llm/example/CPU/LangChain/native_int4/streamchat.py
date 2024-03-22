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

from ipex_llm.langchain.llms import *
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main(args):
    
    question = args.question
    model_path = args.model_path
    model_family = args.model_family
    n_threads = args.thread_num
    template ="""{question}"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    model_family_to_llm = {
        "llama": LlamaLLM,
        "gptneox": GptneoxLLM,
        "bloom": BloomLLM,
        "starcoder": StarcoderLLM,
        "chatglm": ChatGLMLLM
    }

    if model_family in model_family_to_llm:
        langchain_llm = model_family_to_llm[model_family]
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    
    # Verbose is required to pass to the callback manager
    llm = langchain_llm(
        model_path=model_path,
        n_threads=n_threads,
        callback_manager=callback_manager, 
        verbose=True
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    llm_chain.run(question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDLCausalLM Langchain Streaming Chat Example')
    parser.add_argument('-x','--model-family', type=str, required=True,
                        choices=["llama", "bloom", "gptneox", "chatglm", "starcoder"],
                        help='the model family')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to the converted llm model')
    parser.add_argument('-q', '--question', type=str, default='What is AI?',
                        help='qustion you want to ask.')
    parser.add_argument('-t','--thread-num', type=int, default=2,
                        help='Number of threads to use for inference')
    args = parser.parse_args()
    
    main(args)
