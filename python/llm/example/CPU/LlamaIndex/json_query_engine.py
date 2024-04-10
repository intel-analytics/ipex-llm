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
from llama_index.core.indices.struct_store import JSONQueryEngine
import llama_index.core.indices.struct_store.json_query as json_query_module
from ipex_llm.llamaindex.llms import IpexLLM
import argparse

def custom_default_output_processor(llm_output, json_value):
    """Default output processor that extracts values based on JSON Path expressions."""
    # Post-process the LLM output to remove the JSONPath: prefix
    llm_output = llm_output.replace("JSONPath: ", "").replace("JSON Path: ", "").replace("JSON Path query: ", "").strip()    
    # Split the given string into separate JSON Path expressions
    expression = [expr.strip() for expr in llm_output.split("\n\n")][0]
    try:
        from jsonpath_ng.ext import parse
        from jsonpath_ng.jsonpath import DatumInContext
    except ImportError as exc:
        IMPORT_ERROR_MSG = "You need to install jsonpath-ng to use this function!"
        raise ImportError(IMPORT_ERROR_MSG) from exc
    results = {}
    try:
        datum: List[DatumInContext] = parse(expression).find(json_value)
        if datum:
            key = expression.split(".")[
                -1
            ]  # Extracting "title" from "$.title", for example
            results[key] = ", ".join(str(i.value) for i in datum)
    except Exception as exc:
        raise ValueError(f"Invalid JSON Path: {expression}") from exc
    return results

def define_JSON_data():
    # Test on some sample data
    json_value = {
        "blogPosts": [
            {
                "id": 1,
                "title": "First blog post",
                "content": "This is my first blog post",
            },
            {
                "id": 2,
                "title": "Second blog post",
                "content": "This is my second blog post",
            },
        ],
        "comments": [
            {
                "id": 1,
                "content": "Nice post!",
                "username": "jerry",
                "blogPostId": 1,
            },
            {
                "id": 2,
                "content": "Interesting thoughts",
                "username": "simon",
                "blogPostId": 2,
            },
            {
                "id": 3,
                "content": "Loved reading this!",
                "username": "simon",
                "blogPostId": 2,
            },
        ],
    }

    # JSON Schema object that the above JSON value conforms to
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": "Schema for a very simple blog post app",
        "type": "object",
        "properties": {
            "blogPosts": {
                "description": "List of blog posts",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "description": "Unique identifier for the blog post",
                            "type": "integer",
                        },
                        "title": {
                            "description": "Title of the blog post",
                            "type": "string",
                        },
                        "content": {
                            "description": "Content of the blog post",
                            "type": "string",
                        },
                    },
                    "required": ["id", "title", "content"],
                },
            },
            "comments": {
                "description": "List of comments on blog posts",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "description": "Unique identifier for the comment",
                            "type": "integer",
                        },
                        "content": {
                            "description": "Content of the comment",
                            "type": "string",
                        },
                        "username": {
                            "description": (
                                "Username of the commenter (lowercased)"
                            ),
                            "type": "string",
                        },
                        "blogPostId": {
                            "description": (
                                "Identifier for the blog post to which the comment"
                                " belongs"
                            ),
                            "type": "integer",
                        },
                    },
                    "required": ["id", "content", "username", "blogPostId"],
                },
            },
        },
        "required": ["blogPosts", "comments"],
    }
    return json_value,  json_schema

def main(args):
    json_value,  json_schema = define_JSON_data()
    llm = IpexLLM(
        model_name=args.model_path,
        tokenizer_name=args.model_path,
        context_window=512,
        max_new_tokens=args.n_predict,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        device_map="cpu",
        )
    nl_query_engine = JSONQueryEngine(
        json_value=json_value,
        json_schema=json_schema,
        llm=llm,
    )
    nl_response = nl_query_engine.query(args.question)

    print("nl_response: ", nl_response)

if __name__ == "__main__":
    json_query_module.default_output_processor = custom_default_output_processor

    parser = argparse.ArgumentParser(description='LlamaIndex IpexLLM Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What comments has simon been writing?',
                        help='qustion you want to ask.')
    parser.add_argument('-n','--n-predict', type=int, default=128,
                        help='max number of predict tokens')
    args = parser.parse_args()
    
    main(args)