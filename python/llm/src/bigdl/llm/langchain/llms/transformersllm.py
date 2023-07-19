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

# This file is adapted from
# https://github.com/hwchase17/langchain/blob/master/langchain/llms/huggingface_pipeline.py

# The MIT License

# Copyright (c) Harrison Chase

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"


class TransformersLLM(LLM):
    """Wrapper around the BigDL-LLM Transformer-INT4 model

    Example:
        .. code-block:: python

            from langchain.llms import TransformersLLM
            llm = TransformersLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    model: Any  #: :meta private:
    """BigDL-LLM Transformer-INT4 model."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct object from model_id"""
        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
                # AutoModelForSeq2SeqLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        # TODO: may refactore this code in the future
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)
        except:
            model = AutoModel.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @classmethod
    def from_model_id_low_bit(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct object from model_id"""
        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        # TODO: may refactore this code in the future
        try:
            model = AutoModelForCausalLM.load_low_bit(model_id, **_model_kwargs)
        except:
            model = AutoModel.load_low_bit(model_id, **_model_kwargs)

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "BigDL-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, **kwargs)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt) :]
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
