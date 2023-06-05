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

from bigdl.llm.langchain.llms import BigdlLLM
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


template ="""{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager


# Make sure the model path is correct for your system!
llm = BigdlLLM(
    model_path="model/ggml/gpt4all-model-q4_0.bin", 
    # model_path="models/ggml/vicuna-model-q4_0.bin",
    callback_manager=callback_manager, 
    verbose=True
)


llm_chain = LLMChain(prompt=prompt, llm=llm)

#question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

question = "What is AI?"
llm_chain.run(question)