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


from transformers import PreTrainedTokenizer, LogitsProcessor
from ipex_llm.utils.modules import insert_fake_module

insert_fake_module("xgrammar.kernels.apply_token_bitmask_inplace_cuda")
insert_fake_module("xgrammar.kernels.apply_token_bitmask_inplace_triton")
insert_fake_module(
    "xgrammar.kernels.apply_token_bitmask_inplace_cuda.apply_token_bitmask_inplace_cuda"
)
insert_fake_module(
    "xgrammar.kernels.apply_token_bitmask_inplace_triton.apply_token_bitmask_inplace_triton"
)

import xgrammar as xgr


def create_json_logits_processor(tokenizer: PreTrainedTokenizer, vocab_size: int, schema=None):
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    if schema is None:
        compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
    else:
        compiled_grammar = grammar_compiler.compile_json_schema(schema)
    processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    return processor


def reset_json_logits_processor(processor: LogitsProcessor) -> LogitsProcessor:
    compiled_grammar = processor.compiled_grammar
    new_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    return new_processor
