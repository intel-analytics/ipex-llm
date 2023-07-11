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
# https://github.com/hwchase17/langchain/blob/master/langchain/embeddings/llamacpp.py

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

"""Wrapper around BigdlLLM embedding models."""
from typing import Any, Dict, List, Optional
import numpy as np

from pydantic import BaseModel, Extra, Field

from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "gpt2"


class TransformersEmbeddings(BaseModel, Embeddings):
    """Wrapper around bigdl-llm transformers embedding models.

    To use, you should have the ``transformers`` python package installed.

    Example:
        .. code-block:: python

            from bigdl.llm.langchain.embeddings import TransformersEmbeddings
            embeddings = TransformersEmbeddings.from_model_id(model_id)
    """

    model: Any  #: :meta private:
    tokenizer: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_NAME
    """Model id to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Construct object from model_id"""
        try:
            from bigdl.llm.transformers import AutoModel
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

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    def embed(self, text: str):
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")  # shape: [1, T]
        embeddings = self.model(input_ids, return_dict=False)[0]  # shape: [1, T, N]
        embeddings = embeddings.squeeze(0).detach().numpy()
        embeddings = np.mean(embeddings, axis=0)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = [self.embed(text, **self.encode_kwargs).tolist() for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a bigdl-llm transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.embed(text, **self.encode_kwargs)
        return embedding.tolist()
