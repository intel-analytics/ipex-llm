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
from transformers.configuration_utils import PretrainedConfig
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

        subfolder = kwargs.get("subfolder", "")
        variant = kwargs.get("variant", None)
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
            if len(args) == 0 else args[0]

        # For huggingface transformers cls.HF_Model.from_pretrained could only restore the model
        # in the original format, which is not quantized,
        # we can convert the model to quantized later.
        model = None

        # Read bigdl_transformers_int4 from config.json
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)

        bigdl_transformers_int4 = config_dict.pop("bigdl_transformers_int4", False)
        if bigdl_transformers_int4:
            # Avoid KeyError
            kwargs["ignore_mismatched_sizes"] = True

        model = cls.HF_Model.from_pretrained(*args, **kwargs)
        print("Note: If there are warnings about mismatched during the loading process, "
              "please ignore them as it is part of the normal flow. "
              "The model will be reconverted to the format of BigDL after loading.")

        # Note that the ggml_matmul_src1_x_src0_t operation cannot currently
        # be recorded in AutoConfig,
        # and this operation is not included in the core Hugging Face infrastructure.
        if bigdl_transformers_int4:
            from .convert import ggml_convert_int4
            # We forcefully modify the model's definition
            # and the tensor shape of int4 weights without quantization.
            model = ggml_convert_int4(model, convert_shape_only=True)
            # Load the quantized model at last.
            archive_file = extract_local_archive_file(pretrained_model_name_or_path,
                                                      subfolder,
                                                      variant)
            state_dict = load_state_dict(archive_file)
            load(model, state_dict)
            del state_dict
        elif load_in_4bit:
            from .convert import ggml_convert_int4
            model = model.to("cpu")
            model = ggml_convert_int4(model)
            model.config.update({"bigdl_transformers_int4": True})

        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel
