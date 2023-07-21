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


def save_low_bit(self, *args, **kwargs):
    invalidInputError(self.config.to_dict().get("bigdl_transformers_low_bit", False),
                      f"Detected this model is not a low-bit model, please use from_pretrained's"
                      f" load_in_4bit or load_in_low_bit parameter to load a 4-bit model first.")
    self.save_pretrained(*args, **kwargs)


class _BaseAutoModelClass:

    HF_MODEL = None

    @classmethod
    def from_pretrained(cls,
                        *args,
                        **kwargs):
        """
        Load a model from a directory or the HF Hub. Use load_in_4bit or load_in_low_bit parameter
        the weight of model's linears can be loaded to low-bit format, like int4, int5 and int8.

        Two new arguments are added to extend Hugging Face's from_pretrained method as follows:
        New Arguments:
            load_in_4bit: boolean value, True means load linear's weight to symmetric int 4.
            load_in_low_bit: str value, options are sym_int4, asym_int4, sym_int5, asym_int5
                             or sym_int8. (sym_int4 means symmetric int 4, asym_int4 means
                             asymmetric int 4, etc.). Relevant low bit optimizations will
                             be applied to the model.
        """
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
            if len(args) == 0 else args[0]
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        bigdl_transformers_low_bit = config_dict.pop("bigdl_transformers_low_bit", False)
        invalidInputError(not bigdl_transformers_low_bit,
                          f"Detected model is a low-bit({bigdl_transformers_low_bit}) model, "
                          f"Please use load_low_bit to load this model.")

        # For huggingface transformers cls.HF_Model.from_pretrained could only restore the model
        # in the original format, which is not quantized,
        # we can convert the model to quantized later.
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_low_bit = kwargs.pop("load_in_low_bit", None)

        if load_in_4bit or load_in_low_bit:
            # load int x-bit
            kwargs["low_cpu_mem_usage"] = True
            # set default torch_dtype='auto'
            kwargs["torch_dtype"] = kwargs.get("torch_dtype", 'auto')
            q_k = load_in_low_bit if load_in_low_bit else "sym_int4"
            model = cls.load_convert(q_k, *args, **kwargs)
        else:
            # load default
            model = cls.HF_Model.from_pretrained(*args, **kwargs)

        return model

    @classmethod
    def load_convert(cls, q_k, *args, **kwargs):
        from .convert import ggml_convert_quant
        invalidInputError(q_k in ggml_tensor_qtype,
                          f"Unknown load_in_low_bit value: {q_k}, expected:"
                          f" sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")
        qtype = ggml_tensor_qtype[q_k]
        model = cls.HF_Model.from_pretrained(*args, **kwargs)
        model = model.to("cpu")
        model = ggml_convert_quant(model, qtype)
        model.config.update({"bigdl_transformers_low_bit": q_k})

        # add save_low_bit to pretrained model dynamically
        import types
        model.save_low_bit = types.MethodType(save_low_bit, model)

        return model

    @classmethod
    def load_low_bit(cls,
                     *args,
                     **kwargs):
        # Read bigdl_transformers_low_bit from config.json
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
            if len(args) == 0 else args[0]
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        bigdl_transformers_low_bit = config_dict.pop("bigdl_transformers_low_bit", False)

        invalidInputError(bigdl_transformers_low_bit,
                          "Detect this model is not a low-bit model, Please use from_pretrained"
                          " with load_in_4bit or load_in_low_bit to get a low-bit model , and "
                          " serialize the model using save_low_bit first.")

        invalidInputError(bigdl_transformers_low_bit in ggml_tensor_qtype,
                          f"Unknown bigdl_transformers_low_bit value: {bigdl_transformers_low_bit},"
                          f" expected: sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")

        # Speed up when loading model
        kwargs["low_cpu_mem_usage"] = True

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

        # add save_low_bit to pretrained model dynamically
        import types
        model.save_low_bit = types.MethodType(save_low_bit, model)

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

        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSpeechSeq2Seq


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSeq2SeqLM
