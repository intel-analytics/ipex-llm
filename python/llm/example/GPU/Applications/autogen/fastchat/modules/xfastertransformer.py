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
# Copyright 2023 The FastChat team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from dataclasses import dataclass
import sys


@dataclass
class XftConfig:
    max_seq_len: int = 4096
    beam_width: int = 1
    eos_token_id: int = -1
    pad_token_id: int = -1
    num_return_sequences: int = 1
    is_encoder_decoder: bool = False
    padding: bool = True
    early_stopping: bool = False
    data_type: str = "bf16_fp16"


class XftModel:
    def __init__(self, xft_model, xft_config):
        self.model = xft_model
        self.config = xft_config


def load_xft_model(model_path, xft_config: XftConfig):
    try:
        import xfastertransformer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Failed to load xFasterTransformer. {e}")
        sys.exit(-1)

    if xft_config.data_type is None or xft_config.data_type == "":
        data_type = "bf16_fp16"
    else:
        data_type = xft_config.data_type
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, padding_side="left", trust_remote_code=True
    )
    xft_model = xfastertransformer.AutoModel.from_pretrained(
        model_path, dtype=data_type
    )
    model = XftModel(xft_model=xft_model, xft_config=xft_config)
    if model.model.rank > 0:
        while True:
            model.model.generate()
    return model, tokenizer
