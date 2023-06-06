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


from typing import Optional, Union, Sequence, List
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.ggml.model.gptneox import gptneox_cpp


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation

    Pass custom parameter values to 'generate' .
    """
    def generate(
        self,
        inputs: Union[Optional[Sequence[int]], Sequence[gptneox_cpp.gptneox_token]]=None,
        max_new_tokens: int = 128,
        top_k: int = 40,
        top_p: float = 0.95,
        temperature: float = 0.80,
        repetition_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        stop: Optional[Union[str, List[str]]]=[],
        **kwargs,
    ) -> Union[Optional[Sequence[int]], Optional[Sequence[gptneox_cpp.gptneox_token]], None]:
        # TODO: modify docs
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95,
            >>>                             temp=1.0, repeat_penalty=1.1):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        # TODO: stop & max_token
        self._generate(tokens=inputs,
                       top_k=top_k,
                       top_p=top_p,
                       temp=temperature,
                       repeat_penalty=repetition_penalty,
                       reset=reset,
                       frequency_penalty=frequency_penalty,
                       presence_penalty=presence_penalty,
                       tfs_z=tfs_z,
                       mirostat_mode=mirostat_mode,
                       mirostat_tau=mirostat_tau,
                       mirostat_eta=mirostat_eta,
                       **kwargs)
