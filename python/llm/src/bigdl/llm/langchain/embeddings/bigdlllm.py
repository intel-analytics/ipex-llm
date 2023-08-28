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

"""Wrapper around BigdlNative embedding models."""
import logging
import importlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.embeddings.base import Embeddings
from .transformersembeddings import TransformersEmbeddings


class BigdlNativeEmbeddings(BaseModel, Embeddings):
    """Wrapper around bigdl-llm embedding models.

    Example:
        .. code-block:: python

            from bigdl.llm.langchain.embeddings import BigdlNativeEmbeddings
            llama = BigdlNativeEmbeddings(model_path="/path/to/model.bin")
    """

    logging.warning("BigdlNativeEmbeddings has been deprecated, "
                    "please switch to the new LMEmbeddings API for sepcific models.")

    model_family: str = "llama"
    """The model family: currently supports llama, gptneox, bloom and starcoder."""

    family_info = {
        'llama': {'module': "bigdl.llm.models", 'class': "Llama"},
        'bloom': {'module': "bigdl.llm.models", 'class': "Bloom"},
        'gptneox': {'module': "bigdl.llm.models", 'class': "Gptneox"},
        'starcoder': {'module':"bigdl.llm.models", 'class': "Starcoder"},
    }  #: :meta private:
    """Info necessary for different model family initiation and configure."""

    client: Any  #: :meta private:
    """The actual model."""

    model_path: str  # TODO: missing doc
    """Path to the converted BigDL-LLM optimized ggml binary checkpoint."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(2, alias="n_threads")
    """Number of threads to use."""

    n_batch: Optional[int] = Field(512, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(0, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that bigdl-llm library is installed."""
        model_path = values["model_path"]
        model_param_names = [
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]
            
        model_family = values["model_family"].lower()
        if model_family not in list(values["family_info"].keys()):
            raise ValueError("Model family '%s' is not supported. Valid" \
                    " values are %s" % (values["model_family"],
                    ','.join(list(values["family_info"].keys()))))

        try:

            b_info = values["family_info"][model_family]
            module = importlib.import_module(b_info['module'])
            class_ = getattr(module, b_info['class'])

            values["client"] = class_(model_path, embedding=True, **model_params)

            # from bigdl.llm.ggml.model.llama import Llama

            # values["client"] = Llama(model_path, embedding=True, **model_params)

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import bigdl-llm library. "
                "Please install the bigdl-llm library to "
                "use this embedding model: pip install bigdl-llm"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load model from path: {model_path}. "
                f"Please make sure the model family {model_family} matches "
                "the model you want to load."
                f"Received error {e}"
            )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)
        return list(map(float, embedding))


class _BaseEmbeddings(BaseModel, Embeddings):
    """Wrapper around bigdl-llm embedding models.

    param model_path: If running with ``native int4``, the path should be converted BigDL-LLM
          optimized ggml binary checkpoint, which should be converted by ``bigdl.llm.llm_convert``.
          If running with ``transformers int4``, the path should be the huggingface repo id
          to be downloaded or the huggingface checkpoint folder.

    Example:
        .. code-block:: python

            from bigdl.llm.langchain.embeddings import LlamaEmbeddings
            llama = LlamaEmbeddings(model_path="/path/to/model.bin")
    """

    ggml_model: str = None
    ggml_module: str = None

    native: bool = True
    """Load model to either BigDL-LLM optimized Transformers or Native (ggml) int4."""

    client: Any  #: :meta private:
    """The actual model."""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the Transformers model."""

    encode_kwargs: Optional[dict] = None
    """Key word arguments to pass when calling the `encode` method of the Transformers model."""

    kwargs: Any
    """Additional key word arguments passed to TransformersLLM."""

    model_path: str
    """Path to the loading model file.
    If native, the path shoule be converted BigDL-LLM optimized ggml binary checkpoint.
    If transformers, the path should be the huggingface repo id to be downloaded
    or the huggingface checkpoint folder."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""\

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(2, alias="n_threads")
    """Number of threads to use."""

    n_batch: Optional[int] = Field(512, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(0, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that bigdl-llm library is installed."""

        native = values["native"]
        model_path = values["model_path"]
        model_kwargs = values["model_kwargs"]
        kwargs = values["kwargs"]
        model_param_names = [
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        try:

            module = importlib.import_module(values["ggml_module"])
            class_ = getattr(module, values["ggml_model"])

            if native:
                values["client"] = class_(model_path, embedding=True, **model_params)
            else:
                kwargs = {} if kwargs is None else kwargs
                values["client"] = TransformersEmbeddings.from_model_id(model_path, model_kwargs,
                                                                        **kwargs)

            # from bigdl.llm.ggml.model.llama import Llama

            # values["client"] = Llama(model_path, embedding=True, **model_params)

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import bigdl-llm library. "
                "Please install the bigdl-llm library to "
                "use this embedding model: pip install bigdl-llm"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load model from path: {model_path}. "
                f"Please make sure the model embedding class matches "
                "the model you want to load."
                f"Received error {e}"
            )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the optimized int4 model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.native:
            embeddings = [self.client.embed(text) for text in texts]
            return [list(map(float, e)) for e in embeddings]
        else:
            return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the optimized int4 model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        if self.native:
            embedding = self.client.embed(text)
            return list(map(float, embedding))
        else:
            return self.client.embed_query(text)


class LlamaEmbeddings(_BaseEmbeddings):
    ggml_model = "Llama"
    ggml_module = "bigdl.llm.models"


class BloomEmbeddings(_BaseEmbeddings):
    ggml_model = "Bloom"
    ggml_module = "bigdl.llm.models"


class GptneoxEmbeddings(_BaseEmbeddings):
    ggml_model = "Gptneox"
    ggml_module = "bigdl.llm.models"


class StarcoderEmbeddings(_BaseEmbeddings):
    ggml_model = "Starcoder"
    ggml_module = "bigdl.llm.models"
