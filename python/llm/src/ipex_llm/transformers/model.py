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

import copy
import torch
import warnings
import transformers
from typing import List
from unittest.mock import patch
from transformers.configuration_utils import PretrainedConfig

from ipex_llm.ggml.quantize import ggml_tensor_qtype, gguf_mixed_qtype
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.gguf.api import load_gguf_model

from .utils import logger, load_state_dict
from .utils import extract_local_archive_file, get_local_shard_files, load_imatrix_data
from .patches import patch_flash_attn_import, patch_sdpa_available

patched_training_mode = None


def save_low_bit(self, *args, **kwargs):
    invalidInputError(self.config.to_dict().get("bigdl_transformers_low_bit", False),
                      f"Detected this model is not a low-bit model, please use from_pretrained's"
                      f" load_in_4bit or load_in_low_bit parameter to load a 4-bit model first.")
    if hasattr(self.config, "quantization_config"):
        delattr(self.config, "quantization_config")
        delattr(self.config, "_pre_quantization_dtype")

    origin_device = self.device
    self.to('cpu')

    kwargs['safe_serialization'] = False

    architectures = getattr(self.config, "architectures", None)
    model_type = getattr(self.config, "model_type", None)
    disk_embedding = getattr(self.config, "bigdl_disk_embedding", False)

    if disk_embedding:
        from ipex_llm.transformers.embedding import DiskEmbedding
        self.apply(DiskEmbedding.restore_normal_embedding)
        self.save_pretrained(*args, **kwargs)
        self.apply(DiskEmbedding.replace_normal_embedding)
    else:
        self.save_pretrained(*args, **kwargs)

    if architectures:
        self.config.update({"architectures": architectures})
    if model_type:
        self.config.update({"model_type": model_type})

    self.config.save_pretrained(args[0])
    if self.can_generate():
        self.generation_config.save_pretrained(args[0])

    import json
    import os
    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    load_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    with open(os.path.join(args[0], "load_keys.json"), "w") as json_file:
        json.dump(load_keys, json_file)
    if origin_device != 'cpu':
        self.to(origin_device)


def _load_pre():
    from transformers import GPTJModel
    from ipex_llm.transformers.models.gptj import gptj_model_new_init
    GPTJModel.__init__ = gptj_model_new_init


class _BaseAutoModelClass:
    HF_MODEL = None

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    @patch("transformers.modeling_utils.is_torch_sdpa_available", patch_sdpa_available, create=True)
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
                             Default to be ``False``.
        :param load_in_low_bit: str value, options are ``'sym_int4'``, ``'asym_int4'``,
                                ``'sym_int5'``, ``'asym_int5'``, ``'sym_int8'``, ``'nf3'``,
                                ``'nf4'``, ``'fp4'``, ``'fp8'``, ``'fp8_e4m3'``, ``'fp8_e5m2'``,
                                ``'fp6'``, ``'gguf_iq2_xxs'``, ``'gguf_iq2_xs'``,
                                ``'gguf_iq1_s'``, ``'gguf_q4k_m'``, ``'gguf_q4k_s'``,
                                ``'fp16'``, ``'bf16'``, ``'fp6_k'``,
                                ``'sym_int4'`` means symmetric int 4, ``'asym_int4'`` means
                                asymmetric int 4, ``'nf4'`` means 4-bit NormalFloat, etc.
                                Relevant low bit optimizations will be applied to the model.
        :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
                               Default to be ``True``.
        :param modules_to_not_convert: list of str value, modules (nn.Module) that are skipped when
                                       conducting model optimizations. Default to be ``None``.
        :param speculative: boolean value, Whether to use speculative decoding.
                            Default to be ``False``.
        :param cpu_embedding: Whether to replace the Embedding layer, may need to set it
            to ``True`` when running BigDL-LLM on GPU on Windows. Default to be ``False``.
        :param disk_embedding: Whether to put the Embedding layer on disk to save memory.
            Default to be ``False``.
        :param imatrix: str value, represent filename of importance matrix pretrained on
            specific datasets for use with the improved quantization methods recently
            added to llama.cpp.
        :param model_hub: str value, options are ``'huggingface'`` and ``'modelscope'``,
            specify the model hub. Default to be ``'huggingface'``.
        :param embedding_qtype: str value, options are ``'q2_k'``, ``'q4_k'`` now.
            Default to be None. Relevant low bit optimizations will be applied to
            ``nn.Embedding`` layer.
        :param mixed_precision: boolean value, Whether to use mixed precision quantization.
            Default to be False. If set to True, we will use sym_int8 for lm_head when
            load_in_low_bit is sym_int4 or asym_int4.
        :param pipeline_parallel_stages: int value, the number of GPUs allocated for
            pipeline parallel. Default to be ``1``. Please set pipeline_parallel_stages > 1
            to run pipeline parallel inference on multiple GPUs.
        :return: a model instance
        """
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None) \
            if len(args) == 0 else args[0]
        model_hub = kwargs.pop("model_hub", "huggingface")
        invalidInputError(model_hub in ["huggingface", "modelscope"],
                          "The parameter `model_hub` is supposed to be `huggingface` or "
                          f"`modelscope`, but got {model_hub}.")
        invalidInputError(not (kwargs.get('device_map') and 'xpu' in kwargs['device_map']),
                          "Please do not use `device_map` "
                          "with `xpu` value as an argument. "
                          "Use model.to('xpu') instead.")
        if model_hub == "huggingface":
            config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        elif model_hub == "modelscope":
            import modelscope
            from modelscope.utils.hf_util import get_wrapped_class
            cls.HF_Model = get_wrapped_class(cls.HF_Model)
            from .utils import get_modelscope_hf_config
            config_dict, _ = get_modelscope_hf_config(pretrained_model_name_or_path)
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
        speculative = kwargs.pop("speculative", False)
        pipeline_parallel_stages = kwargs.pop("pipeline_parallel_stages", 1)
        torch_dtype = kwargs.pop("torch_dtype", None)
        embedding_qtype = kwargs.pop("embedding_qtype", None)

        if user_quantization_config is not None and \
                "BitsAndBytesConfig" in str(user_quantization_config.__class__):
            if user_quantization_config.bnb_4bit_quant_type is not None:
                bnb_4bit_type = user_quantization_config.bnb_4bit_quant_type
                if bnb_4bit_type == "nf4":
                    load_in_low_bit = "nf4"
                elif bnb_4bit_type == "fp4":
                    warnings.warn(
                        "BigDL LLM QLoRA does not support fp4 now, use default nf4", FutureWarning)
                    load_in_low_bit = "nf4"
                elif bnb_4bit_type == "int4":
                    load_in_low_bit = "sym_int4"
                elif bnb_4bit_type == "bf16":
                    load_in_low_bit = "bf16"
                else:
                    invalidInputError(False,
                                      "Only nf4 or int4 is supported for bnb_4bit_quant_type")
            else:
                warnings.warn(
                    "bnb_4bit_quant_type is None, use default int4", FutureWarning)
                load_in_low_bit = "sym_int4"
            if user_quantization_config.bnb_4bit_use_double_quant is True:
                warnings.warn(
                    "BigDL LLM QLoRA does not support double quant now, set to False",
                    FutureWarning)
            if user_quantization_config.bnb_4bit_compute_dtype is not None:
                bnb_dtype = user_quantization_config.bnb_4bit_compute_dtype
                if bnb_dtype == torch.float32:
                    kwargs["torch_dtype"] = bnb_dtype
                elif bnb_dtype == torch.bfloat16:
                    kwargs["torch_dtype"] = bnb_dtype
                else:
                    invalidInputError(False,
                                      "Only float32 or bfloat16"
                                      " is supported for bnb_4bit_compute_dtype")
            else:
                warnings.warn(
                    "torch_dtype is None, use default float32", FutureWarning)
                kwargs["torch_dtype"] = torch.float32
            optimize_model = False
            kwargs["modules_to_not_convert"] = ["lm_head"]

        load_in_8bit = kwargs.pop("load_in_8bit", False)
        from ipex_llm.llm_patching import bigdl_patched
        if bigdl_patched == 'Train':
            global patched_training_mode
            if load_in_low_bit == "nf4" or load_in_low_bit == "sym_int4" or load_in_4bit:
                # qlora
                patched_training_mode = 'qlora'
            else:
                # lora
                patched_training_mode = 'lora'
                load_in_low_bit = "bf16"
            optimize_model = False
            kwargs["modules_to_not_convert"] = ["lm_head"]

        if load_in_4bit or load_in_low_bit:

            if config_dict.get("quantization_config", None) is not None:
                from ipex_llm.transformers.low_bit_linear import get_block_size
                q_config = config_dict["quantization_config"]
                if q_config["quant_method"] == "gptq":
                    invalidInputError(q_config["bits"] == 4,
                                      "Only 4-bit gptq is supported in bigdl-llm.")
                    if load_in_low_bit is not None:
                        invalidInputError(load_in_low_bit == "asym_int4",
                                          "You can only load gptq model as aysm_int4 low bit type.")

                    load_in_low_bit = "asym_int4"
                    if int(q_config["group_size"]) % get_block_size(load_in_low_bit) != 0:
                        invalidInputError(False,
                                          (f"group_size must be divisible by "
                                           f"{get_block_size(load_in_low_bit)}."))
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
                    from ipex_llm.transformers.awq.awq_config import AwqConfig
                    awq_config = AwqConfig.from_dict(q_config)
                    invalidInputError(awq_config.bits == 4,
                                      "Only 4-bit awq is supported in bigdl-llm.")
                    invalidInputError(awq_config.version == "gemm",
                                      "Only gemm version is supported in bigdl-llm.")
                    invalidInputError(awq_config.zero_point,
                                      "Only awq zero_point = True is supported in bigdl-llm.")
                    if load_in_low_bit is not None:
                        invalidInputError(load_in_low_bit == "asym_int4",
                                          "You can only load awq model as aysm_int4 low bit type.")

                    load_in_low_bit = "asym_int4"

                    if int(awq_config.group_size) % get_block_size(load_in_low_bit) != 0:
                        invalidInputError(False,
                                          (f"group_size must be divisible by "
                                           f"{get_block_size(load_in_low_bit)}."))

                    kwargs["quantization_config"] = awq_config

            # load int x-bit
            kwargs["low_cpu_mem_usage"] = True
            # set default torch_dtype='auto'.
            # Note that when load_in_low_bit="fp16", set default torch_dtype=torch.float16
            if load_in_low_bit == "fp16":
                if torch_dtype is not None and torch_dtype != torch.float16:
                    invalidInputError(
                        False,
                        f"Please use torch_dtype=torch.float16"
                        f" when setting load_in_low_bit='fp16'."
                    )
                else:
                    kwargs["torch_dtype"] = torch.float16
            elif load_in_low_bit == "bf16":
                if torch_dtype is not None and torch_dtype != torch.bfloat16:
                    invalidInputError(
                        False,
                        f"Please use torch_dtype=torch.bfloat16"
                        f" when setting load_in_low_bit='bf16'."
                    )
                else:
                    kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch_dtype or "auto"
            # Avoid tensor parallel F.Linear Operations
            if "pretraining_tp" in config_dict:
                if "config" in kwargs:
                    setattr(kwargs["config"], "pretraining_tp", 1)
                else:
                    kwargs["pretraining_tp"] = 1
            q_k = load_in_low_bit if load_in_low_bit else "sym_int4"

            invalidInputError(q_k not in ["sym_int4_rtn", "sym_int8_rtn"],
                              f"The dtype {q_k} is specified for NPU"
                              "and cannot be used on CPU and GPU")

            imatrix_file = kwargs.pop("imatrix", None)
            if q_k in ["gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s"]:
                invalidInputError(imatrix_file is not None,
                                  "For gguf_iq2 and gguf_iq1 quantization,"
                                  "imatrix is needed.")
            cpu_embedding = kwargs.get("cpu_embedding", False)
            # for iq2/k-quants, default use embedding_quantization
            if not cpu_embedding and embedding_qtype is None:
                if q_k in ["gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "q2_k"]:
                    embedding_qtype = "q2_k"
                elif q_k in ["gguf_q4k_s", "gguf_q4k_m"]:
                    embedding_qtype = "q4_k"
            if imatrix_file is not None:
                imatrix_data = load_imatrix_data(imatrix_file)
                kwargs["imatrix_data"] = imatrix_data
            kwargs["embedding_qtype"] = embedding_qtype
            model = cls.load_convert(q_k, optimize_model, *args, **kwargs)

            if pipeline_parallel_stages > 1:
                if speculative:
                    invalidInputError(False,
                                      f"Please do not set speculative=True"
                                      f" when using pipeline_parallel_stages")
                invalidInputError(torch.distributed.get_world_size() == pipeline_parallel_stages,
                                  "Please make sure you've called `init_pipeline_parallel()` "
                                  "and world size is the same as `pipeline_parallel_stages`")
                from .pipeline_parallel import pipeline_parallel, pipeline_parallel_generate
                model = pipeline_parallel(model, pipeline_parallel_stages, kwargs["torch_dtype"])
                import types
                # add pipeline_parallel_generate to pretrained model dynamically
                model.pipeline_parallel_generate = types.MethodType(pipeline_parallel_generate,
                                                                    model)
                torch.distributed.barrier()
            if speculative:
                from .speculative import speculative_generate, clear_benchmarks,\
                    _crop_past_key_values
                # load a sym_int4 model as draft model
                draft_model = cls.load_convert('sym_int4', optimize_model, *args, **kwargs)
                model.draft_model = draft_model
                import types
                # add speculative_generate to pretrained model dynamically
                model.clear_benchmarks = types.MethodType(clear_benchmarks, model)
                model.speculative_generate = types.MethodType(speculative_generate, model)
                model._crop_past_key_values = types.MethodType(_crop_past_key_values, model)

            # add lookup_generate to pretrained model
            from .lookup import lookup_generate
            import types
            model.lookup_generate = types.MethodType(lookup_generate, model)
            if model.config.model_type == "minicpmv" and hasattr(model, 'llm'):
                model.llm.lookup_generate = types.MethodType(lookup_generate, model.llm)
        else:
            # load default
            model = cls.HF_Model.from_pretrained(*args, **kwargs)

        return model

    @staticmethod
    def from_gguf(fpath: str, optimize_model: bool = True,
                  cpu_embedding: bool = False, low_bit: str = "sym_int4"):
        """
        Load gguf model and tokenizer and convert it to bigdl-llm model and huggingface tokenzier

        :param fpath: Path to gguf model file
        :param optimize_model: Whether to further optimize llm model, defaults to True
        :param cpu_embedding: Whether to replace the Embedding layer, may need to set it
            to `True` when running BigDL-LLM on GPU on Windows, defaults to False

        :return: An optimized bigdl-llm model and a huggingface tokenizer
        """
        from ipex_llm.optimize import optimize_model as optimize_model_fn

        model, tokenizer = load_gguf_model(fpath, dtype=torch.half, low_bit=low_bit)
        model = optimize_model_fn(model, low_bit=low_bit, optimize_llm=optimize_model,
                                  cpu_embedding=cpu_embedding)
        return model, tokenizer

    @classmethod
    def load_convert(cls, q_k, optimize_model, *args, **kwargs):
        from .convert import ggml_convert_low_bit
        invalidInputError(q_k in ggml_tensor_qtype or q_k in gguf_mixed_qtype,
                          f"Unknown load_in_low_bit value: {q_k}, expected:"
                          f" sym_int4, asym_int4, sym_int5, asym_int5, sym_int8, nf3, nf4, "
                          f"fp4, fp6, fp8, fp8_e4m3, fp8_e5m2, fp16,  bf16, gguf_iq2_xxs, "
                          f"gguf_iq2_xs, gguf_iq1_s, q2_k, q4_k, q5_k, q6_k, fp6_k"
                          f"gguf_q4k_s, gguf_q4k_m, mixed_fp4 or mixed_fp8.")
        if q_k in ggml_tensor_qtype:
            qtype = ggml_tensor_qtype[q_k]
        else:
            qtype = gguf_mixed_qtype[q_k]

        # In case it needs a second try,
        # `from_pretrained`` may pop items out in dict
        # and lead to args missing.
        modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        cpu_embedding = kwargs.pop("cpu_embedding", False)
        if kwargs.pop("replace_embedding", False):
            warnings.warn("replace_embedding is deprecated and will be removed in a future version,"
                          " please use cpu_embedding instead.", FutureWarning)
            cpu_embedding = True
        disk_embedding = kwargs.pop("disk_embedding", False)
        quant_config = kwargs.pop("quantization_config", None)
        imatrix_data = kwargs.pop("imatrix_data", None)
        embedding_qtype = kwargs.pop("embedding_qtype", None)
        mixed_precision = kwargs.pop("mixed_precision", False)
        if embedding_qtype is not None:
            embedding_qtype = ggml_tensor_qtype[embedding_qtype]
        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
        awq_config = None

        if quant_config and quant_config.quant_method == "awq":
            # The latest transformers only support cuda version
            # This load awq ckpt logic is copied from
            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/models/base.py#L147
            from accelerate import init_empty_weights, infer_auto_device_map, \
                load_checkpoint_in_model
            from ipex_llm.transformers.awq.awq import _replace_with_awq_layers, \
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
            if quant_config is not None:
                kwargs["quantization_config"] = quant_config
            _load_pre()
            try:
                # To handle the input CUDA setting (such as 'device_map={"":0}'), ignore it
                kwargs.pop('device_map', None)
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
                                     cpu_embedding=cpu_embedding,
                                     torch_dtype=kwargs.get("torch_dtype", 'auto'),
                                     imatrix_data=imatrix_data,
                                     embedding_qtype=embedding_qtype,
                                     mixed_precision=mixed_precision)

        if disk_embedding:
            from ipex_llm.transformers.embedding import DiskEmbedding
            model.apply(DiskEmbedding.replace_normal_embedding)

        model.config.update({"bigdl_transformers_low_bit": q_k,
                             "bigdl_disk_embedding": disk_embedding})

        # enable tie_word_embeddings for MPT
        # refer to https://huggingface.co/mosaicml/mpt-7b-chat/blob/main/modeling_mpt.py#L232
        if model.config.architectures is None \
           or model.config.architectures[0] != 'MPTForCausalLM':
            model.config.update({"tie_word_embeddings": False})

        # add save_low_bit to pretrained model dynamically
        import types
        model.save_low_bit = types.MethodType(save_low_bit, model)

        return model

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    @patch("transformers.modeling_utils.is_torch_sdpa_available", patch_sdpa_available, create=True)
    def load_low_bit(cls,
                     pretrained_model_name_or_path,
                     *model_args,
                     **kwargs):
        """
        Load a low bit optimized model (including INT4, INT5 and INT8) from a saved ckpt.

        :param pretrained_model_name_or_path: str value, Path to load the optimized model ckpt.
        :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
                               Default to be True.
        :param pipeline_parallel_stages: int value, the number of GPUs allocated for
            pipeline parallel. Default to be ``1``. Please set pipeline_parallel_stages > 1
            to run pipeline parallel inference on multiple GPUs.

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
        cpu_embedding = kwargs.pop("cpu_embedding", False)
        if kwargs.pop("replace_embedding", False):
            warnings.warn("replace_embedding is deprecated and will be removed in a future version,"
                          " please use cpu_embedding instead.", FutureWarning)
            cpu_embedding = True
        disk_embedding = kwargs.pop("disk_embedding", False)
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
        embedding_qtype = kwargs.pop("embedding_qtype", None)
        sharded_metadata = None

        pipeline_parallel_stages = kwargs.pop("pipeline_parallel_stages", 1)

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
        bigdl_transformers_low_bit = config_dict.pop("bigdl_transformers_low_bit", False)
        bigdl_lcmu_enabled = config_dict.pop("bigdl_lcmu_enabled", True)

        invalidInputError(bigdl_transformers_low_bit,
                          "Detect this model is not a low-bit model, Please use from_pretrained"
                          " with load_in_4bit or load_in_low_bit to get a low-bit model , and "
                          " serialize the model using save_low_bit first.")

        invalidInputError(bigdl_transformers_low_bit in ggml_tensor_qtype or
                          bigdl_transformers_low_bit in gguf_mixed_qtype,
                          f"Unknown bigdl_transformers_low_bit value: {bigdl_transformers_low_bit},"
                          f" expected: sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")

        # set default optimize_model=True
        optimize_model = kwargs.pop("optimize_model", True)

        if bigdl_transformers_low_bit in ggml_tensor_qtype:
            qtype = ggml_tensor_qtype[bigdl_transformers_low_bit]
        else:
            qtype = gguf_mixed_qtype[bigdl_transformers_low_bit]
        if bigdl_transformers_low_bit in ["gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq1_s", "q2_k"] and \
                not cpu_embedding:
            embedding_qtype = "q2_k"
        elif bigdl_transformers_low_bit in ["gguf_q4k_s", "gguf_q4k_m"] and \
                not cpu_embedding:
            embedding_qtype = "q4_k"
        if embedding_qtype is not None:
            embedding_qtype = ggml_tensor_qtype[embedding_qtype]

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
                if config.architectures is not None and config.architectures[0] in \
                   ["ChatGLMModel", "ChatGLMForConditionalGeneration"]:

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
        model = ggml_convert_low_bit(model, qtype, optimize_model, device=quant_device,
                                     modules_to_not_convert=modules_to_not_convert,
                                     cpu_embedding=cpu_embedding,
                                     embedding_qtype=embedding_qtype, torch_dtype=torch_dtype)

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
            _fast_init=False,  # always false to avoid pre-init behaviors
            low_cpu_mem_usage=bigdl_lcmu_enabled,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            keep_in_fp32_modules=[],
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        if disk_embedding:
            from ipex_llm.transformers.embedding import DiskEmbedding
            model.apply(DiskEmbedding.replace_normal_embedding)
            model.config.update({"bigdl_disk_embedding": disk_embedding})

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

        # rwkv model linear layers has been rescaled
        if model.config.model_type == "rwkv":
            model.rwkv.layers_are_rescaled = True

        if pipeline_parallel_stages > 1:
            from .pipeline_parallel import pipeline_parallel, pipeline_parallel_generate
            model = pipeline_parallel(model, pipeline_parallel_stages, torch_dtype)
            import types
            # add pipeline_parallel_generate to pretrained model dynamically
            model.pipeline_parallel_generate = types.MethodType(pipeline_parallel_generate,
                                                                model)
            torch.distributed.barrier()

        # add lookup_generate to loaded model
        from .lookup import lookup_generate
        import types
        model.lookup_generate = types.MethodType(lookup_generate, model)
        if model.config.model_type == "minicpmv" and hasattr(model, 'llm'):
            model.llm.lookup_generate = types.MethodType(lookup_generate, model.llm)

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
