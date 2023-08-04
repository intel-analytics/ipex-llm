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


from typing import List
from pathlib import Path

from bigdl.llm.libs.chatglm_C import Pipeline, GenerationConfig


class ChatGLMContext:
    def __init__(self, pipeline: Pipeline, config: GenerationConfig):
        self.pipeline = pipeline
        self.config = config


def chatglm_load(path: str,
                 n_ctx: int,
                 n_threads: int,
                 use_mmap: bool = False,
                 ) -> ChatGLMContext:
    path = str(Path(path))
    pipeline = Pipeline(path, use_mmap)
    config = GenerationConfig(
        max_context_length=n_ctx,
        num_threads=n_threads,
    )
    return ChatGLMContext(pipeline, config)


def chatglm_tokenize(ctx: ChatGLMContext, prompt: str) -> List[int]:
    return ctx.pipeline.tokenizer.encode(prompt)


def chatglm_detokenize(ctx: ChatGLMContext, input_ids: List[int]) -> str:
    return ctx.pipeline.tokenizer.decode(input_ids)


def chatglm_eval(ctx: ChatGLMContext,
                 input_ids: List[int],
                 n_past: int,
                 do_sample: bool = True,
                 top_k: int = 0,
                 top_p: float = 0.7,
                 temperature: float = 0.95,
                 ) -> int:
    ctx.config.do_sample = do_sample
    ctx.config.top_k = top_k
    ctx.config.top_p = top_p
    ctx.temperature = temperature
    return ctx.pipeline.model.generate_next_token(input_ids, ctx.config, n_past,
                                                  ctx.config.max_context_length)


def chatglm_eos_token(ctx: ChatGLMContext):
    return ctx.pipeline.model.config.eos_token_id
