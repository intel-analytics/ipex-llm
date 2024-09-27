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

import os
import copy
import types
import warnings
import torch
import transformers
from typing import List
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers.configuration_utils import PretrainedConfig

from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.utils import logger
from ipex_llm.transformers.npu_models.convert import optimize_llm, optimize_llm_post


def patch_flash_attn_import(filename: str) -> List[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def ignore_argument(kwargs: dict, key: "str"):
    arg = kwargs.pop(key, None)
    if arg is not None:
        warnings.warn(f"argument `{key}={arg}` will be ignored")


def save_low_bit(self, model_dir: str, *args, **kwargs):
    origin_device = self.device
    kwargs["safe_serialization"] = False
    self.save_pretrained(model_dir, *args, **kwargs)
    import json

    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    load_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    with open(os.path.join(model_dir, "load_keys.json"), "w") as json_file:
        json.dump(load_keys, json_file)
    if origin_device != "cpu":
        self.to(origin_device)


class _BaseAutoModelClass:
    HF_MODEL = None

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def from_pretrained(cls, *args, **kwargs):
        """
        Load a model from a directory or the HF Hub. Use load_in_low_bit parameter to convert
        model to low-bit format, like int4 and int8.
        The loaded model will run supported OPs on NPU, then run other OPs on CPU.

        Three new arguments are added to extend Hugging Face's from_pretrained method as follows:
        :param load_in_low_bit: str value, options are ``'sym_int4'``, ``'sym_int8'``,
                                ``'fp16'``, ``'fp32'``.
                                Relevant low bit optimizations will be applied to the model.
        :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
                               Default to be ``False``.
        :param mixed_precision: boolean value, Whether to use mixed precision quantization.
            Default to be False. If set to ``True``, we will use ``'sym_int8'`` for lm_head when
            ``load_in_low_bit`` is '``sym_int4``' for certain models.
        :return: a model instance
        """
        if kwargs.get("device_map", None) not in [None, "cpu", "auto"]:
            warnings.warn("`device_map` will be ignored")
        kwargs["device_map"] = "cpu"

        if kwargs.get("torch_dtype", None) not in [None, "auto", torch.float, torch.float16]:
            warnings.warn("`torch_dtype` will be ignored, `torch.float` will be used")
            kwargs["torch_dtype"] = torch.float32

        low_bit = kwargs.pop("load_in_low_bit", "sym_int4")
        qtype_map = {
            "sym_int4": "sym_int4_rtn",
            "sym_int8": "sym_int8_rtn",
        }

        invalidInputError(
            low_bit in qtype_map.keys(),
            f"unsupported low_bit: {low_bit}, " f"only {list(qtype_map.keys())} are supported",
        )
        qtype = qtype_map[low_bit]

        kwargs["low_cpu_mem_usage"] = True

        # ignore following arguments
        ignore_argument(kwargs, "model_hub")
        ignore_argument(kwargs, "lightweight_bmm")
        ignore_argument(kwargs, "load_in_4bit")
        ignore_argument(kwargs, "load_in_8bit")
        ignore_argument(kwargs, "imatrix")
        ignore_argument(kwargs, "cpu_embedding")
        ignore_argument(kwargs, "embedding_qtype")
        ignore_argument(kwargs, "enable_mp")
        ignore_argument(kwargs, "quantization_config")
        ignore_argument(kwargs, "speculative")
        ignore_argument(kwargs, "pipeline_parallel_stages")
        optimize_model = kwargs.pop("optimize_model", False)
        max_output_len = kwargs.pop("max_output_len", 1024)
        max_output_len = max_output_len - 1
        max_prompt_len = kwargs.pop("max_prompt_len", 512)
        inter_pp = kwargs.pop("inter_pp", None)
        intra_pp = kwargs.pop("intra_pp", None)
        transpose_value_cache = kwargs.pop("transpose_value_cache", True)
        modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        mixed_precision = kwargs.pop('mixed_precision', False)

        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
        try:
            # To handle the input CUDA setting (such as 'device_map={"":0}'), ignore it
            kwargs.pop("device_map", None)
            model = cls.HF_Model.from_pretrained(*args, **kwargs)
        except NotImplementedError:
            logger.info(
                "Failed to load models with `low_cpu_mem_usage` specified, "
                "will fall to traditional load method with higher memory consumption."
            )
            _kwargs["low_cpu_mem_usage"] = False
            model = cls.HF_Model.from_pretrained(*_args, **_kwargs)
            model.config.update({"bigdl_lcmu_enabled": False})

        logger.info(f"Converting model, it may takes up to several minutes ...")
        from intel_npu_acceleration_library.compiler import create_npu_kernels

        if optimize_model:
            invalidInputError(
                max_prompt_len < max_output_len,
                (
                    f"max_prompt_len ({max_prompt_len}) should be less"
                    " than max_output_len ({max_output_len})"
                ),
            )
            from ipex_llm.transformers.npu_models.convert_mp import optimize_llm, optimize_llm_pre

            if hasattr(model, "llm"):
                llm = model.llm
            else:
                llm = model

            with torch.no_grad():
                model.config.update({"mixed_precision": mixed_precision})
                optimize_llm_pre(model, qtype, mixed_precision)
                cls.load_convert(qtype, model, "cpu", modules_to_not_convert, *args, **kwargs)
                create_npu_kernels(llm)
            model = model.eval()
            logger.info(f"Finish to convert model")
            model.config.update({"bigdl_transformers_low_bit": qtype})
            model.share_memory()

            optimize_llm(
                llm,
                max_output_len=max_output_len,
                max_prompt_len=max_prompt_len,
                inter_pp=inter_pp,
                intra_pp=intra_pp,
                transpose_value_cache=transpose_value_cache,
            )
            model.save_low_bit = types.MethodType(save_low_bit, model)
        else:
            from ipex_llm.transformers.npu_models.convert import optimize_llm
            optimize_llm(model)
            with torch.no_grad():
                cls.load_convert(qtype, model, "cpu", modules_to_not_convert, *args, **kwargs)
                if hasattr(model, "llm"):
                    create_npu_kernels(model.llm)
                else:
                    create_npu_kernels(model)
            model = model.eval()
            logger.info(f"Finish to convert model")
            model.config.update({"bigdl_transformers_low_bit": qtype})
            # add save_low_bit to pretrained model dynamically
            model.save_low_bit = types.MethodType(save_low_bit, model)

        return model

    @classmethod
    def load_convert(cls, q_k, optimize_model, device, modules_to_not_convert, *arg, **kwarg):
        from ipex_llm.transformers.npu_models.convert import replace_with_QuantizedLinear

        replace_with_QuantizedLinear(optimize_model, q_k, device=device,
                                     modules_to_not_convert=modules_to_not_convert)

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def load_low_bit(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        # ignore following arguments
        ignore_argument(kwargs, "model_hub")
        ignore_argument(kwargs, "lightweight_bmm")
        ignore_argument(kwargs, "cpu_embedding")
        ignore_argument(kwargs, "embedding_qtype")
        ignore_argument(kwargs, "speculative")
        ignore_argument(kwargs, "pipeline_parallel_stages")
        ignore_argument(kwargs, "mixed_precision")
        optimize_model = kwargs.pop("optimize_model", False)
        max_output_len = kwargs.pop("max_output_len", 1024)
        max_prompt_len = kwargs.pop("max_prompt_len", 512)
        inter_pp = kwargs.pop("inter_pp", None)
        intra_pp = kwargs.pop("intra_pp", None)
        transpose_value_cache = kwargs.pop("transpose_value_cache", True)
        modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])

        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.modeling_utils import no_init_weights, get_state_dict_dtype
        from transformers.dynamic_module_utils import (
            resolve_trust_remote_code,
            get_class_from_dynamic_module,
        )
        from transformers.models.auto.auto_factory import _get_model_class
        from transformers.utils.generic import ContextManagers
        from transformers.generation.configuration_utils import GenerationConfig
        from ipex_llm.transformers.utils import (
            extract_local_archive_file,
            get_local_shard_files,
            load_state_dict,
        )
        from accelerate.big_modeling import init_empty_weights

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
        qtype = config_dict.pop("bigdl_transformers_low_bit", False)
        bigdl_lcmu_enabled = config_dict.pop("bigdl_lcmu_enabled", True)
        mixed_precision = config_dict.pop("mixed_precision", False)

        invalidInputError(
            qtype,
            "Detect this model is not a low-bit model, Please use from_pretrained"
            " with load_in_4bit or load_in_low_bit to get a low-bit model , and "
            " serialize the model using save_low_bit first.",
        )

        invalidInputError(
            qtype in ["sym_int8_rtn", "sym_int4_rtn"],
            f"Unknown bigdl_transformers_low_bit value: {qtype},"
            f" expected: sym_int8_rtn, sym_int4_rtn. "
        )

        has_remote_code = hasattr(config, "auto_map") and cls.HF_Model.__name__ in config.auto_map
        has_local_code = type(config) in cls.HF_Model._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.HF_Model.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.HF_Model.__name__)
            else:
                cls.HF_Model.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in cls.HF_Model._model_mapping.keys():
            model_class = _get_model_class(config, cls.HF_Model._model_mapping)

        resolved_archive_file, is_sharded = extract_local_archive_file(
            pretrained_model_name_or_path, subfolder, variant
        )

        if is_sharded:
            resolved_archive_file, sharded_metadata = get_local_shard_files(
                pretrained_model_name_or_path, resolved_archive_file, subfolder=subfolder
            )

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
                    invalidInputError(
                        False,
                        f'`torch_dtype` can be either `torch.dtype` or `"auto"`,'
                        "but received {torch_dtype}",
                    )
            dtype_orig = model_class._set_default_torch_dtype(torch_dtype)

        # Pretrained Model
        _fast_init = kwargs.pop("_fast_init", True)
        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())

        if bigdl_lcmu_enabled:
            with ContextManagers(init_contexts):
                if config.architectures is not None and config.architectures[0] in [
                    "ChatGLMModel",
                    "ChatGLMForConditionalGeneration",
                ]:

                    """
                    ChatGLMModel uses skip_init by default, which will force modules placed on cpu
                    if the device is not specified. This will further cause replaced linear
                    allocating memory on cpu.
                    """
                    kwargs["device"] = "meta"
                model = model_class(config, *model_args, **kwargs)
        else:
            model = model_class(config, *model_args, **kwargs)

        # Loading args may differ based on their usage
        quant_device = "meta" if bigdl_lcmu_enabled else "cpu"
        logger.info(f"Converting model, it may takes up to several minutes ...")
        from intel_npu_acceleration_library.compiler import create_npu_kernels

        if optimize_model:
            invalidInputError(
                max_prompt_len < max_output_len,
                (
                    f"max_prompt_len ({max_prompt_len}) should be less"
                    " than max_output_len ({max_output_len})"
                ),
            )
            from ipex_llm.transformers.npu_models.convert_mp import optimize_llm_pre

            if hasattr(model, "llm"):
                llm = model.llm
            else:
                llm = model

            with torch.no_grad():
                optimize_llm_pre(model, qtype, mixed_precision)
                cls.load_convert(qtype, model, quant_device, modules_to_not_convert,
                                 *model_args, **kwargs)
                create_npu_kernels(llm)

        else:
            from ipex_llm.transformers.npu_models.convert import optimize_llm
            optimize_llm(model)
            with torch.no_grad():
                cls.load_convert(qtype, model, quant_device, modules_to_not_convert,
                                 *model_args, **kwargs)
                create_npu_kernels(model)

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            import json

            with open(
                os.path.join(pretrained_model_name_or_path, "load_keys.json"), "r"
            ) as json_file:
                loaded_data = json.load(json_file)
            loaded_state_dict_keys = loaded_data["all_checkpoint_keys"]

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        # set tie_word_embeddings to False to avoid possible lm_head error
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False

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
            _fast_init=False,  # always false to avoid pre-init behaviors
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

        if optimize_model:
            from ipex_llm.transformers.npu_models.convert_mp import optimize_llm
            optimize_llm(
                llm,
                max_output_len=max_output_len,
                max_prompt_len=max_prompt_len,
                inter_pp=inter_pp,
                intra_pp=intra_pp,
                transpose_value_cache=transpose_value_cache,
            )

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
