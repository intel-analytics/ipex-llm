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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/callbacks.py


import gc
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers
from transformers import is_torch_xpu_available

import modules.shared as shared


class StopNowException(Exception):
    pass


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return shared.stop_everything


class StopWordsCriteria(transformers.StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, stop_words, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        texts =  [self.tokenizer.decode(input_ids[-1][-1])]
        dones = [any(stop_word in texts for stop_word in self.stop_words)]
        return all(dones)


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])

        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """

    def __init__(self, func, args=None, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.args = args or []
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now or shared.stop_everything:
                raise StopNowException
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, *args, **self.kwargs)
            except StopNowException:
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()
