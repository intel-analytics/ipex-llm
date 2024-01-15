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

import torch
from bigdl.llm.utils.common import invalidInputError


qtype_map = {
    2: "sym_int4",      # q4_0
    3: "asym_int4",     # q4_1
    7: "sym_int8",      # q8_0
    8: "sym_int5",      # q5_0
    9: "asym_int5",     # q5_1
}


def load_gguf_model(fpath: str, dtype: torch.dtype = torch.float):
    from .gguf import GGUFFileLoader

    loader = GGUFFileLoader(fpath)
    model_family = loader.config["general.architecture"]
    print("model_family:" + model_family)
    qtype = loader.config["general.file_type"]

    invalidInputError(qtype in qtype_map, f"Unsupported gguf quantize type: {qtype}")
    low_bit = qtype_map.get(qtype, "sym_int4")

    with torch.no_grad():
        if model_family == "llama":
            general_name = loader.config["general.name"].lower()
            if "mixtral" in general_name:
                # mixtral, which also enjoys a general architecture of llama
                from .models.mixtral import load_gguf_mixtral
                model, tokenizer = load_gguf_mixtral(loader, dtype)
            elif "mistral" in general_name:
                from .models.mistral import load_gguf_mistral
                model, tokenizer = load_gguf_mistral(loader, dtype)
            else:
                from .models.llama import load_gguf_llama
                model, tokenizer = load_gguf_llama(loader, dtype)
        elif model_family == "baichuan":
            from .models.baichuan import load_gguf_baichuan
            model, tokenizer = load_gguf_baichuan(loader, dtype)
        elif model_family == "bloom":
            from .models.bloom import load_gguf_bloom
            model, tokenizer = load_gguf_bloom(loader, dtype)
        elif model_family == "falcon":
            from .models.falcon import load_gguf_falcon
            model, tokenizer = load_gguf_falcon(loader, dtype)
        elif model_family == "mpt":
            from .models.mpt import load_gguf_mpt
            model, tokenizer = load_gguf_mpt(loader, dtype)
        else:
            invalidInputError(False, f"Unsupported model family: {model_family}")

        return model, tokenizer, low_bit
