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
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError


class _BaseAutoModelClass:

    HF_MODEL = None

    @classmethod
    def from_pretrained(cls,
                        *args,
                        **kwargs):

        # For huggingface transformers cls.HF_Model.from_pretrained could only restore the model
        # in the original format, which is not quantized,
        # we can convert the model to quantized later.
        model = None
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_low_bit = kwargs.pop("load_in_low_bit", None)

        # Read bigdl_transformers_low_bit from config.json
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
            if len(args) == 0 else args[0]
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        bigdl_transformers_low_bit = config_dict.pop("bigdl_transformers_low_bit", False)

        if load_in_4bit or load_in_low_bit or bigdl_transformers_low_bit:
            # Speed up when loading model
            kwargs["low_cpu_mem_usage"] = True

        if bigdl_transformers_low_bit:
            invalidInputError(bigdl_transformers_low_bit in ggml_tensor_qtype,
                              f"Unknown load_in_low_bit value: {bigdl_transformers_low_bit},"
                              f" excepted q4_0, q4_1, q5_0, q5_1, q8_0.")
            qtype = ggml_tensor_qtype[bigdl_transformers_low_bit]
            # Note that the int4 linear layers cannot currently
            # be recorded in huggingface Pretrained Model or AutoConfig,
            # and huggingface transformers cls.HF_Model.from_pretrained
            # could only restore the model in the original format,
            # which is not quantized. we can Initialize original model first,
            # convert the model to quantized int4 format later, and then load the quantized model.

            # Avoid KeyError
            kwargs["ignore_mismatched_sizes"] = True
            # Avoid reading from local file at the first initialization
            kwargs["state_dict"] = {}

            # Maybe needed when extract_local_archive_file
            subfolder = kwargs.get("subfolder", "")
            variant = kwargs.get("variant", None)

            from .convert import ggml_convert_quant
            model = cls.HF_Model.from_pretrained(*args, **kwargs)
            print("Note: If there are warnings during the model loading process, "
                  "they can be safely ignored; "
                  "the model will be loaded with INT4 optimizations applied.")

            # We forcefully modify the model's definition
            # and the tensor shape of int4 weights without quantization.
            model = ggml_convert_quant(model, qtype, convert_shape_only=True)
            # Load the quantized model at last.
            archive_file = extract_local_archive_file(pretrained_model_name_or_path,
                                                      subfolder,
                                                      variant)
            state_dict = load_state_dict(archive_file)
            load(model, state_dict)
            del state_dict

        elif load_in_4bit or load_in_low_bit:
            q_k = load_in_low_bit if load_in_low_bit else "q4_0"
            model = cls.convert_quant(model, q_k, *args, **kwargs)

        return model

    @classmethod
    def convert_quant(cls, model, q_k, *args, **kwargs):
        from .convert import ggml_convert_quant
        invalidInputError(q_k in ggml_tensor_qtype,
                          f"Unknown load_in_low_bit value: {q_k},"
                          f" excepted q4_0, q4_1, q5_0, q5_1, q8_0.")
        qtype = ggml_tensor_qtype[q_k]
        model = cls.HF_Model.from_pretrained(*args, **kwargs)
        model = model.to("cpu")
        model = ggml_convert_quant(model, qtype)
        model.config.update({"bigdl_transformers_low_bit": q_k})
        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel
