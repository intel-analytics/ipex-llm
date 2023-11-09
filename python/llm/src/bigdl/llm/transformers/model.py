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
from .utils import extract_local_archive_file, \
    load_state_dict, \
    get_local_shard_files
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError
import torch
from bigdl.llm.utils.lazy_load_torch import LazyLoadTensors
import copy
import inspect
from functools import partial
from accelerate.utils import (
    offload_weight,
    set_module_tensor_to_device,
)
from .utils import logger

from .convert import ggml_convert_low_bit
from bigdl.llm.transformers.low_bit_linear import LowBitLinear, ggml_convert_qtype


class HFLoadMetaModelManager:
    def __init__(self, q_k="sym_int4"):
        self.q_k = q_k
        self.old_loadmethod = transformers.modeling_utils._load_state_dict_into_meta_model

    def __enter__(self):
        _stream_load = partial(_stream_load_meta_model, q_k=self.q_k)
        transformers.modeling_utils._load_state_dict_into_meta_model = _stream_load

    def __exit__(self, exc_type, exc_value, traceback):
        transformers.modeling_utils._load_state_dict_into_meta_model = self.old_loadmethod


def save_low_bit(self, *args, **kwargs):
    invalidInputError(self.config.to_dict().get("bigdl_transformers_low_bit", False),
                      f"Detected this model is not a low-bit model, please use from_pretrained's"
                      f" load_in_4bit or load_in_low_bit parameter to load a 4-bit model first.")
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


def _stream_load_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    device_map=None,
    offload_folder=None,
    offload_index=None,
    state_dict_folder=None,
    state_dict_index=None,
    dtype=None,
    is_quantized=False,
    is_safetensors=False,
    keep_in_fp32_modules=None,
    q_k = "sym_int4",
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # XXX: remaining features to implement to be fully compatible with _load_state_dict_into_model
    # - deepspeed zero 3 support
    # - need to copy metadata if any - see _load_state_dict_into_model
    # - handling error_msgs - mimicking the error handling in module._load_from_state_dict()
    # - Is there a situation where some keys aren't in `loaded_state_dict_keys` and in which case
    #   they won't get loaded.

    if is_quantized:
        from transformers.integrations import set_module_quantized_tensor_to_device

    error_msgs = []

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    
    # Todo: wrap this with function tools
    qtype = ggml_tensor_qtype[q_k]
    if not hasattr(model, "bigdl_meta_convert"):
        # quant/convert linear to QX_X in meta device
        model = ggml_convert_low_bit(model, qtype, True, device="meta")
        model.bigdl_meta_convert = True

    for param_name, param_ld in state_dict.items():
        # First part of the test is always true as load_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        module_name = param_name
        set_module_kwargs = {}

        # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # in int/uint/bool and not cast them.
        if dtype is not None and torch.is_floating_point(param_ld):
            if (
                keep_in_fp32_modules is not None
                and any(module_to_keep_in_fp32 in param_name for module_to_keep_in_fp32 in keep_in_fp32_modules)
                and dtype == torch.float16
            ):
                param_ld = param_ld.to(torch.float32)

                # For backward compatibility with older versions of `accelerate`
                # TODO: @sgugger replace this check with version check at the next `accelerate` release
                if "dtype" in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs["dtype"] = torch.float32
            else:
                param_ld = param_ld.to(dtype)

        param = None
        # The pickled data will be actually loaded into RAM here.
        param_ld = param_ld.load()
        if "." in param_name:
            splits = param_name.split(".")
            module = model
            for split in splits[:-1]:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f"{module} has no attribute {split}.")
                module = new_module
            if isinstance(module, LowBitLinear) and splits[-1] == 'weight':
                # quantize its param if this param is corresponding to LowBitLinear
                param = ggml_convert_qtype(param_ld.contiguous().float(), qtype,
                                           device="cpu")
            else:
                param = param_ld


        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
        if dtype is None:
            old_param = model
            splits = param_name.split(".")
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break

            if old_param is not None:
                param = param.to(old_param.dtype)

        set_module_kwargs["value"] = param

        if device_map is None:
            param_device = "cpu"
        else:
            # find next higher level module that is defined in device_map:
            # bert.lm_head.weight -> bert.lm_head -> bert -> ''
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif not is_quantized:
            # For backward compatibility with older versions of `accelerate`
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            if param.dtype == torch.int8 and param_name.replace("weight", "SCB") in state_dict.keys():
                fp16_statistics = state_dict[param_name.replace("weight", "SCB")]
            else:
                fp16_statistics = None

            if "SCB" not in param_name:
                set_module_quantized_tensor_to_device(
                    model, param_name, param_device, value=param, fp16_statistics=fp16_statistics
                )
    return error_msgs, offload_index, state_dict_index


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

        :param load_in_4bit: boolean value, True means load linear's weight to symmetric int 4.
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

        if load_in_4bit or load_in_low_bit:
            kwargs["low_cpu_mem_usage"] = True
            # Avoid tensor parallel F.Linear Operations
            if "pretraining_tp" in config_dict:
                if "config" in kwargs:
                    setattr(kwargs["config"], "pretraining_tp", 1)
                else:
                    kwargs["pretraining_tp"] = 1
            q_k = load_in_low_bit if load_in_low_bit else "sym_int4"
            # Try 1xlowbit memory convert first,
            # may occur NotImplementedError when doing layernorm
            try:
                _args = copy.deepcopy(args)
                _kwargs = copy.deepcopy(kwargs)
                with HFLoadMetaModelManager(q_k=q_k), LazyLoadTensors():
                    model = cls.HF_Model.from_pretrained(*_args, **_kwargs)
            except:
                # In with case we backup to traditional convert method
                logger.info("Failed to load models with 1x lowbit peak memory, "
                            "will fall to traditional load method with higher memory consumption.")
                # load int x-bit
                # set default torch_dtype='auto'
                kwargs["torch_dtype"] = kwargs.get("torch_dtype", 'auto')
                model = cls.load_convert(q_k, optimize_model, *args, **kwargs)
            model.to("cpu") # may remove
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
                          "fp4, fp8, fp16 or mixed_4bit.")
        qtype = ggml_tensor_qtype[q_k]

        # In case it needs a second try,
        # `from_pretrained`` may pop items out in dict
        # and lead to args missing.
        modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        replace_embedding = kwargs.pop("replace_embedding", False)
        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
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
