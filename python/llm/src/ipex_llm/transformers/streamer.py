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
# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
#

from typing import Optional, List

import torch
from transformers import TextIteratorStreamer


class BatchTextIteratorStreamer(TextIteratorStreamer):
    """
    A specialized version of TextIteratorStreamer that handles text streams in batches, providing
    an efficient way to process large volumes of text data asynchronously. This class is designed
    to aggregate multiple texts into batches, making it ideal for applications that need to
    perform batch operations on streamed text data, such as bulk text processing or machine
    learning model inference in an interactive environment.

        Parameters:
                tokenizer (`AutoTokenizer`):
                        The tokenized used to decode the tokens.
                skip_prompt (`bool`, *optional*, defaults to `False`):
                        Whether to skip the prompt to `.generate()` or not.
                timeout (`float`, *optional*):
                        The timeout for the text queue. If `None`, the queue will
                        block indefinitely. Useful to handle exceptions
                        in `.generate()`, when it is called in a separate thread.
                decode_kwargs (`dict`, *optional*):
                        Additional keyword arguments to pass to the tokenizer's `decode` method.
                batch_size(`int`)
                        The size of the batches to process. This parameter must be specified and
                        determines how many texts are processed together as a single batch.
    """

    def __init__(
        self,
        batch_size: int,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(
                value, (self.batch_size, value.shape[0] // self.batch_size)
            )

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx]:]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx]:]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx]:text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(
                    self.token_cache[idx], **self.decode_kwargs
                )
                printable_text = text[self.print_len[idx]:]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
