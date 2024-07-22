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
# Adapted from
# https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/types.py


from pydantic import BaseModel, field_validator
from typing import List, Optional
from ipex_llm.utils.common import invalidInputError


class Parameters(BaseModel):
    max_new_tokens: int = 32
    do_sample: Optional[bool] = None
    min_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None

    @field_validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            invalidInputError(False, "`repetition_penalty` must be strictly positive")
        return v

    @field_validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            invalidInputError(False, "`temperature` must be strictly positive")
        return v

    @field_validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            invalidInputError(False, "`top_k` must be strictly positive")
        return v

    @field_validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            invalidInputError(False, "`top_p` must be > 0.0 and < 1.0")
        return v

    @field_validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            invalidInputError(False, "`typical_p` must be > 0.0 and < 1.0")
        return v
