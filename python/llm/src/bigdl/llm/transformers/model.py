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
#
# MIT License
#
# Copyright (c) 2023 MIT HAN Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import transformers
from transformers.configuration_utils import PretrainedConfig
from .utils import extract_local_archive_file, \
    load_state_dict, \
    get_local_shard_files
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError
import torch
import warnings
import copy
from .utils import logger


def save_low_bit(self, *args, **kwargs):
    invalidInputError(self.config.to_dict().get("bigdl_transformers_low_bit", False),
                      f"Detected this model is not a low-bit model, please use from_pretrained's"
                      f" load_in_4bit or load_in_low_bit parameter to load a 4-bit model first.")
    if hasattr(self.config, "quantization_config"):
        delattr(self.config, "quantization_config")
        delattr(self.config, "_pre_quantization_dtype")

    self.to('cpu')
    self.save_pretrained(*args, **kwargs)
    import json
    import os
    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    load_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    with open(os.path.join(args[0], "load_keys.json"), "w") as json_file:
        json.dump(load_keys, json_file)


class _BaseAutoModelClass:

    HF_MODEL = None

    @classmethod
    def from_pretrained(cls,
                        *args,
                        **kwargs):
        """
        Load a model from a directory or the HF Hub. Use load_in_4bit or load_in_low_bit parameter
        the weight of model's linears can be loaded to low-bit format, like int4, int5 and int8.

        Three new arguments are added to extend Hugging Face's from_pretrained method as follows:

        :param load_in_4bit: boolean value, True means loading linear's weight to symmetric int 4 if
                                the model is a regular fp16/bf16/fp32 model, and to asymmetric int 4
                                if the model is GPTQ model.
                             Default to be False.
        :param load_in_low_bit: str value, options are sym_int4, asym_int4, sym_int5, asym_int5
                                , sym_int8, nf3, nf4, fp4, fp8 or fp16. sym_int4 means symmetric
                                 int 4, asym_int4 means asymmetric int 4, nf4 means 4-bit
                                 NormalFloat, etc. Relevant low bit optimizations will be applied
                                 to the model.
        :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
                               Default to be True.
        :param modules_to_not_convert: list of str value, modules (nn.Module) that are skipped when
                                       conducting model optimizations. Default to be None.
        :param replace_embedding: Whether to replace the Embedding layer, may need to set it
            to `True` when running BigDL-LLM on GPU on Windows. Default to be `False`.
        :return: a model instance
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
        optimize_model = kwargs.pop("optimize_model", True)
        user_quantization_config = kwargs.pop("quantization_config", None)

        if load_in_4bit or load_in_low_bit:

            if config_dict.get("quantization_config", None) is not None:
                from bigdl.llm.transformers.low_bit_linear import get_ggml_qk_size
                q_config = config_dict["quantization_config"]
                if q_config["quant_method"] == "gptq":
                    invalidInputError(q_config["bits"] == 4,
                                      "Only 4-bit gptq is supported in bigdl-llm.")
                    invalidInputError(q_config["desc_act"] is False,
                                      "Only desc_act=False is supported in bigdl-llm.")
                    if load_in_low_bit is not None:
                        invalidInputError(load_in_low_bit == "asym_int4",
                                          "You can only load gptq model as aysm_int4 low bit type.")

                    load_in_low_bit = "asym_int4"
                    if int(q_config["group_size"]) % get_ggml_qk_size(load_in_low_bit) != 0:
                        invalidInputError(False,
                                          (f"group_size must be divisible by "
                                           f"{get_ggml_qk_size(load_in_low_bit)}."))
                    if user_quantization_config is not None:
                        invalidInputError(user_quantization_config.bits == 4,
                                          "Only 4-bit gptq is supported in bigdl-llm.")
                        invalidInputError(user_quantization_config.use_exllama is False,
                                          "Only use_exllama=False is supported in bigdl-llm.")
                    else:
                        from transformers import GPTQConfig
                        user_quantization_config = GPTQConfig(bits=4, use_exllama=False)
                    kwargs["quantization_config"] = user_quantization_config
                elif q_config["quant_method"] == "awq":
                    from bigdl.llm.transformers.awq.awq_config import AwqConfig
                    awq_config = AwqConfig.from_dict(q_config)
                    invalidInputError(awq_config.bits == 4,
                                      "Only 4-bit awq is supported in bigdl-llm.")
                    invalidInputError(awq_config.version == "gemm",
                                      "Only gemm version is supported in bigdl-llm.")
                    invalidInputError(awq_config.backend == "autoawq",
                                      "Only autoawq backend is supported in bigdl-llm.")
                    invalidInputError(awq_config.zero_point,
                                      "Only awq zero_point = True is supported in bigdl-llm.")
                    if load_in_low_bit is not None:
                        invalidInputError(load_in_low_bit == "asym_int4",
                                          "You can only load awq model as aysm_int4 low bit type.")

                    load_in_low_bit = "asym_int4"

                    if int(awq_config.group_size) % get_ggml_qk_size(load_in_low_bit) != 0:
                        invalidInputError(False,
                                          (f"group_size must be divisible by "
                                           f"{get_ggml_qk_size(load_in_low_bit)}."))

                    kwargs["quantization_config"] = awq_config

            # load int x-bit
            kwargs["low_cpu_mem_usage"] = True
            # set default torch_dtype='auto'
            kwargs["torch_dtype"] = kwargs.get("torch_dtype", 'auto')
            # Avoid tensor parallel F.Linear Operations
            if "pretraining_tp" in config_dict:
                if "config" in kwargs:
                    setattr(kwargs["config"], "pretraining_tp", 1)
                else:
                    kwargs["pretraining_tp"] = 1
            q_k = load_in_low_bit if load_in_low_bit else "sym_int4"
            model = cls.load_convert(q_k, optimize_model, *args, **kwargs)
        else:
            # load default
            model = cls.HF_Model.from_pretrained(*args, **kwargs)

        return model

    @classmethod
    def load_convert(cls, q_k, optimize_model, *args, **kwargs):
        from .convert import ggml_convert_low_bit
        invalidInputError(q_k in ggml_tensor_qtype,
                          f"Unknown load_in_low_bit value: {q_k}, expected:"
                          f" sym_int4, asym_int4, sym_int5, asym_int5, sym_int8, nf3, nf4, "
                          "fp4, fp8, fp16, mixed_fp4 or mixed_fp8.")
        qtype = ggml_tensor_qtype[q_k]

        # In case it needs a second try,
        # `from_pretrained`` may pop items out in dict
        # and lead to args missing.
        modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        replace_embedding = kwargs.pop("replace_embedding", False)
        quant_config = kwargs.pop("quantization_config", None)
        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
        awq_config = None

        if quant_config and quant_config.quant_method == "awq":
            # The latest transformers only support cuda version
            # This load awq ckpt logic is copied from
            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/models/base.py#L147
            from accelerate import init_empty_weights, infer_auto_device_map,\
                load_checkpoint_in_model
            from bigdl.llm.transformers.awq.awq import _replace_with_awq_layers,\
                get_layer_type, _load_config
            awq_config = quant_config
            model_weights_path, config = _load_config(args[0], '', max_new_tokens=None,
                                                      safetensors=True)
            with init_empty_weights():
                model = cls.HF_Model.from_config(config=config, trust_remote_code=True)

            _replace_with_awq_layers(model, awq_config=awq_config)

            model.tie_weights()

            # Get device map
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=[get_layer_type(config)],
                max_memory=None,
                dtype=config.torch_dtype
            )

            # Load checkpoint
            load_checkpoint_in_model(
                model,
                checkpoint=model_weights_path,
                device_map=device_map,
                offload_folder=None,
                dtype=config.torch_dtype
            )

            # Offloading dispatch
            from accelerate import dispatch_model
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None
            )
        else:
            try:
                model = cls.HF_Model.from_pretrained(*args, **kwargs)
            except NotImplementedError:
                logger.info("Failed to load models with `low_cpu_mem_usage` specified, "
                            "will fall to traditional load method with higher memory consumption.")
                _kwargs["low_cpu_mem_usage"] = False
                model = cls.HF_Model.from_pretrained(*_args, **_kwargs)
                model.config.update({"bigdl_lcmu_enabled": False})

        model = model.to("cpu")
        model = ggml_convert_low_bit(model, qtype, optimize_model,
                                     modules_to_not_convert=modules_to_not_convert,
                                     replace_embedding=replace_embedding)
        model.config.update({"bigdl_transformers_low_bit": q_k})
        model.config.update({"tie_word_embeddings": False})

        # add save_low_bit to pretrained model dynamically
        import types
        model.save_low_bit = types.MethodType(save_low_bit, model)

        return model

    @classmethod
    def load_low_bit(cls,
                     pretrained_model_name_or_path,
                     *model_args,
                     **kwargs):
        """
        Load a low bit optimized model (including INT4, INT5 and INT8) from a saved ckpt.

        :param pretrained_model_name_or_path: str value, Path to load the optimized model ckpt.
        :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
                               Default to be True.

        :return: a model instance
        """
        from transformers.modeling_utils import no_init_weights, get_state_dict_dtype
        from transformers.dynamic_module_utils import resolve_trust_remote_code, \
            get_class_from_dynamic_module
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.utils.generic import ContextManagers
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.models.auto.auto_factory import _get_model_class
        from accelerate.big_modeling import init_empty_weights
        from .convert import ggml_convert_low_bit
        import copy
        import os

        modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        replace_embedding = kwargs.pop("replace_embedding", False)
        # Autofactory
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs_orig = copy.deepcopy(kwargs)

        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # if torch_dtype=auto was passed here, ensure to pass it on
        if kwargs_orig.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"

        # Maybe needed when extract_local_archive_file
        subfolder = kwargs.get("subfolder", "")
        variant = kwargs.get("variant", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        torch_dtype = kwargs.pop("torch_dtype", "auto")
        sharded_metadata = None

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        bigdl_transformers_low_bit = config_dict.pop("bigdl_transformers_low_bit", False)
        bigdl_lcmu_enabled = config_dict.pop("bigdl_lcmu_enabled", True)

        invalidInputError(bigdl_transformers_low_bit,
                          "Detect this model is not a low-bit model, Please use from_pretrained"
                          " with load_in_4bit or load_in_low_bit to get a low-bit model , and "
                          " serialize the model using save_low_bit first.")

        invalidInputError(bigdl_transformers_low_bit in ggml_tensor_qtype,
                          f"Unknown bigdl_transformers_low_bit value: {bigdl_transformers_low_bit},"
                          f" expected: sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")

        # set default optimize_model=True
        optimize_model = kwargs.pop("optimize_model", True)

        qtype = ggml_tensor_qtype[bigdl_transformers_low_bit]

        has_remote_code = hasattr(config, "auto_map") and cls.HF_Model.__name__ in config.auto_map
        has_local_code = type(config) in cls.HF_Model._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.HF_Model.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path,  **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.HF_Model.__name__)
            else:
                cls.HF_Model.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in cls.HF_Model._model_mapping.keys():
            model_class = _get_model_class(config, cls.HF_Model._model_mapping)

        resolved_archive_file, is_sharded = extract_local_archive_file(
            pretrained_model_name_or_path,
            subfolder,
            variant)

        if is_sharded:
            resolved_archive_file, sharded_metadata = \
                get_local_shard_files(pretrained_model_name_or_path,
                                      resolved_archive_file,
                                      subfolder=subfolder)

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict,
        #    by checking its first weights entry that is of a floating type
        #    - we assume all floating dtype weights are of the same dtype
        # we also may have config.torch_dtype available, but we won't rely on it till v5
        dtype_orig = None

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                        torch_dtype = config.torch_dtype

                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file[0])
                            torch_dtype = get_state_dict_dtype(one_state_dict)
                            del one_state_dict  # free CPU memory
                else:
                    invalidInputError(False,
                                      f'`torch_dtype` can be either `torch.dtype` or `"auto"`,'
                                      'but received {torch_dtype}')
            dtype_orig = model_class._set_default_torch_dtype(torch_dtype)

        # Pretrained Model
        _fast_init = kwargs.pop("_fast_init", True)
        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())

        if bigdl_lcmu_enabled:
            with ContextManagers(init_contexts):
                model = model_class(config, *model_args, **kwargs)
        else:
            model = model_class(config, *model_args, **kwargs)

        # Loading args may differ based on their usage
        quant_device = "meta" if bigdl_lcmu_enabled else "cpu"
        model = ggml_convert_low_bit(model, qtype, optimize_model, device=quant_device,
                                     modules_to_not_convert=modules_to_not_convert,
                                     replace_embedding=replace_embedding)

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            import os
            import json
            with open(os.path.join(pretrained_model_name_or_path,
                                   "load_keys.json"), "r") as json_file:
                loaded_data = json.load(json_file)
            loaded_state_dict_keys = loaded_data["all_checkpoint_keys"]

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = model_class._load_pretrained_model(
            model,
            None,
            loaded_state_dict_keys,  # XXX: rename?
            resolved_archive_file,
            pretrained_model_name_or_path,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=bigdl_lcmu_enabled,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            keep_in_fp32_modules=[],
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    **kwargs,
                )
            except (OSError, TypeError):
                pass
        for param in model.parameters():
            param.requires_grad_(False)
        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSpeechSeq2Seq


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSeq2SeqLM


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSequenceClassification


class AutoModelForMaskedLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMaskedLM


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForQuestionAnswering


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForNextSentencePrediction


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMultipleChoice


class AutoModelForTokenClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForTokenClassification
