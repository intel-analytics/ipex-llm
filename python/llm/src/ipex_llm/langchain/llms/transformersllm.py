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
from ipex_llm.utils.common.log4Error import invalidInputError
from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"


class TransformersLLM(LLM):
    """Wrapper around the BigDL-LLM Transformer-INT4 model

    Example:
        .. code-block:: python

            from ipex_llm.langchain.llms import TransformersLLM
            llm = TransformersLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    model: Any  #: :meta private:
    """BigDL-LLM Transformers-INT4 model."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        device_map: str = 'cpu',
        tokenizer_id: str = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct object from model_id

        Args:

            model_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder.
            model_kwargs: Keyword arguments that will be passed to the model and tokenizer.
            kwargs: Extra arguments that will be passed to the model and tokenizer.

        Returns:
            An object of TransformersLLM.
        """
        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
                # AutoModelForSeq2SeqLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            invalidInputError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        if tokenizer_id is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **_model_kwargs)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id, **_model_kwargs)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        # TODO: may refactore this code in the future
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True,
                                                         **_model_kwargs)
        except:
            model = AutoModel.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)

        # TODO: may refactore this code in the future
        if 'xpu' in device_map:
            model = model.to(device_map)

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
        device_map: str = 'cpu',
        tokenizer_id: str = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct low_bit object from model_id
        Args:
            model_id: Path for the bigdl transformers low-bit model checkpoint folder.
            model_kwargs: Keyword arguments that will be passed to the model and tokenizer.
            kwargs: Extra arguments that will be passed to the model and tokenizer.

        Returns:
            An object of TransformersLLM.
        """
        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            invalidInputError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        if tokenizer_id is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **_model_kwargs)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id, **_model_kwargs)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        # TODO: may refactore this code in the future
        try:
            model = AutoModelForCausalLM.load_low_bit(model_id, **_model_kwargs)
        except:
            model = AutoModel.load_low_bit(model_id, **_model_kwargs)
        # TODO: may refactore this code in the future
        if 'xpu' in device_map:
            model = model.to(device_map)

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
        if self.streaming:
            from transformers import TextStreamer
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            if stop is not None:
                from transformers.generation.stopping_criteria import StoppingCriteriaList
                from transformers.tools.agents import StopSequenceCriteria
                # stop generation when stop words are encountered
                # TODO: stop generation when the following one is stop word
                stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop,
                                                                               self.tokenizer)])
            else:
                stopping_criteria = None
            output = self.model.generate(input_ids, streamer=streamer,
                                         stopping_criteria=stopping_criteria, **kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return text
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
            if stop is not None:
                from transformers.generation.stopping_criteria import StoppingCriteriaList
                from transformers.tools.agents import StopSequenceCriteria
                stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop,
                                                                               self.tokenizer)])
            else:
                stopping_criteria = None
            output = self.model.generate(input_ids, stopping_criteria=stopping_criteria, **kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            return text
