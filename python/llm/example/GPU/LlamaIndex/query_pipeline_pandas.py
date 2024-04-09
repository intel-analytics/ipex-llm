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
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.core.query_engine.pandas import PandasInstructionParser
import llama_index.core.query_engine.pandas.output_parser as output_parser_module
from llama_index.core.query_engine.pandas.output_parser import default_output_processor
from llama_index.core import PromptTemplate
from ipex_llm.llamaindex.llms import IpexLLM
import pandas as pd
import numpy as np
import argparse

def custom_default_output_processor(output, df, **output_kwargs):
    output = output.split('\n\n')[0]
    output = default_output_processor(output, df, **output_kwargs)
    return output


def output_parser():
    df = pd.read_csv("./titanic_train.csv")
    pandas_output_parser = PandasInstructionParser(df)
    return df, pandas_output_parser

def define_prompt(df):
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    return pandas_prompt, response_synthesis_prompt

def build_query_pipeline(pandas_prompt, llm, pandas_output_parser, response_synthesis_prompt):
    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    # add link from response synthesis prompt to llm2
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

def main(args):
    llm = IpexLLM(
        model_name=args.model_path,
        tokenizer_name=args.model_path,
        context_window=512,
        max_new_tokens=args.n_predict,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        device_map="xpu",
    )

    df, pandas_output_parser = output_parser()
    pandas_prompt, response_synthesis_prompt = define_prompt(df)
    qp = build_query_pipeline(pandas_prompt, llm, pandas_output_parser, response_synthesis_prompt)

    response = qp.run(
        query_str=args.question,
    )

    print("response: ", response)

if __name__ == "__main__":
    output_parser_module.default_output_processor = custom_default_output_processor

    parser = argparse.ArgumentParser(description='LlamaIndex BigdlLLM Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is the correlation between survival and age?',
                        help='qustion you want to ask.')
    parser.add_argument('-n','--n-predict', type=int, default=128,
                        help='max number of predict tokens')
    args = parser.parse_args()
    
    main(args)