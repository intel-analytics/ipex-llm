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

import transformers
import torch
from .utils import extract_local_archive_file, load_state_dict, load


class _BaseAutoModelClass:

    HF_MODEL = None

    @classmethod
    def from_pretrained(cls,
                        *args,
                        **kwargs):
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        if load_in_4bit:
            kwargs["low_cpu_mem_usage"] = True

        save_convert_pretrained = kwargs.pop("save_convert_pretrained", None)
        load_convert_pretrained = kwargs.pop("load_convert_pretrained", False)
        subfolder = kwargs.get("subfolder", "")
        variant = kwargs.get("variant", None)
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
                                        if len(args) == 0 else args[0]

        model = None
        if load_convert_pretrained:
            kwargs["ignore_mismatched_sizes"] = True    
        model = cls.HF_Model.from_pretrained(*args, **kwargs)
        if load_convert_pretrained:
            from .convert import ggml_convert_int4
            model = ggml_convert_int4(model, convert_shape_only=True)
            archive_file = extract_local_archive_file(pretrained_model_name_or_path, subfolder, variant)
            state_dict = load_state_dict(archive_file)
            load(model, state_dict)
            del state_dict
        elif load_in_4bit:
            from .convert import ggml_convert_int4
            model = model.to("cpu")
            model = ggml_convert_int4(model)
            if save_convert_pretrained:
                print("save_convert_pretrained:", save_convert_pretrained)
                model.save_pretrained(save_convert_pretrained)
                config = model.config
                config.quantized = True
                config.save_pretrained(save_convert_pretrained)

        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel
