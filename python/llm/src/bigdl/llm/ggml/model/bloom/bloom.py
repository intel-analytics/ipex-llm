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
# ===========================================================================
#
# This file is adapted from
# https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py
#
# MIT License
#
# Copyright (c) 2023 Andrei Betlen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

from .bloom_cpp import bloom_load, bloom_free, bloom_run
from bigdl.llm.utils.common import invalidInputError
from typing import List, Optional
import time
import uuid


class Bloom:
    """High-level Python wrapper for a bloom.cpp model."""

    def __init__(self,
                 model_path: str,
                 n_ctx: int = 512,
                 seed: int = 1337,
                 logits_all: bool = False,
                 n_threads: int = -1,
                 n_batch: int = 8,
                 last_n_tokens_size: int = 64,
                 verbose: bool = True,
                 ):
        """Load a bloom.cpp model from `model_path`.

        Args:
            model_path: Path to the model.
            n_ctx: Maximum context size.
            seed: Random seed. 0 for random.
            logits_all: Return logits for all tokens, not just the last token.
            n_threads: Number of threads to use.
                       If None, the number of threads is automatically determined.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Bloom instance.
        """
        self.model_path = model_path
        self.ctx = bloom_load(bytes(model_path, encoding='utf-8'), n_ctx, n_threads)
        invalidInputError(self.ctx is not None, f"Failed to load model from {model_path}")
        self.n_ctx = n_ctx
        self.seed = seed
        self.logits_all = logits_all
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.verbose = verbose

    def __call__(self, prompt: str, max_tokens: int, stream: bool = False,
                 stop: Optional[List[str]] = []):
        if stream:
            return self.stream(prompt, max_tokens, stop)
        else:
            return self._eval(prompt, max_tokens, False, stop)

    def _eval(self, prompt: str, max_tokens: int, match_str: bool,
              stop: Optional[List[str]] = []):
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        if prompt.endswith("</s>") or max_tokens < 1:
            return {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": prompt,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage":
                {
                    # TODO: need tokenize
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
            }
        # use `buf` to store prompt and generated string,
        # assume the average length of words is less than 20 bytes
        buf = bytes((len(prompt) + max_tokens) * 20)
        ret = bloom_run(ctx=self.ctx,
                        seed=self.seed,
                        n_threads=self.n_threads,
                        n_batch=self.n_batch,
                        n_predict=max_tokens,
                        match_str=match_str,
                        prompt=bytes(prompt, encoding='utf-8'),
                        buf=buf)
        s = str(buf, encoding='utf-8').rstrip("\x00")

        text = s.split(prompt)[1]
        split_text = text
        if stop != []:
            for stop_word in stop:
                split_text = split_text.split(stop_word)[0]
        if split_text != text:
            finish_reason = "stop"
        else:
            finish_reason = None
        return {"id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": prompt + split_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage":
                {
                    # TODO: need tokenize
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                }

    def stream(self, prompt: str, max_tokens: int, stop: Optional[List[str]] = []):
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        if prompt.endswith("</s>") or max_tokens < 1:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": prompt,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage":
                    {
                        # TODO: need tokenize
                        "prompt_tokens": None
                }
            }
        else:
            for i in range(max_tokens):
                if prompt.endswith("</s>"):
                    break
                else:
                    prompt = self._eval(prompt, 1, i != 0, stop)['choices'][0]['text']
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": self.model_path,
                        "choices": [
                            {
                                "text": prompt,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                        "usage":
                            {
                                # TODO: need tokenize
                                "prompt_tokens": None
                        }
                    }

    def free(self):
        bloom_free(self.ctx)
