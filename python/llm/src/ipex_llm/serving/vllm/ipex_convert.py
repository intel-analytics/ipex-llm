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
import functools

import torch
import intel_extension_for_pytorch.nn as nn
from torch.ao.quantization import PlaceholderObserver, QConfig, QConfigMapping
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
)

from typing import List, Optional, Tuple, Union

from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
)
from vllm._C import ops
from vllm.logger import init_logger

from ipex_llm.ggml.quantize import ggml_tensor_qtype
from .model_convert import _model_mlp_convert, _model_attention_convert

logger = init_logger(__name__)


def _ipex_convert(model_runner):
    import intel_extension_for_pytorch.quantization
    # import intel_extension_for_pytorch.quantization._utils
    from intel_extension_for_pytorch.nn.modules.weight_only_quantization import IpexWoqLinear
    setattr(intel_extension_for_pytorch.quantization, "convert", _ipex_quantize_convert)
    setattr(intel_extension_for_pytorch.quantization._utils, "module_call_to_function_call",
            _ipex_module_call_to_function_call)
    setattr(IpexWoqLinear, "from_float", _ipex_woq_from_float)

    from .ipex_llm_convert import _ipex_llm_rotary_embedding_forward
    setattr(model_runner, "load_model", _ipex_load_model)
    setattr(RotaryEmbedding, "forward", _ipex_llm_rotary_embedding_forward)
    setattr(RMSNorm, "forward", _ipex_rmsnorm_forward)
    logger.info("The model is optimized with IPEX. ")


def _ipex_load_model(self) -> None:
    _model_mlp_convert()
    _model_attention_convert()

    self.model = get_model(self.model_config,
                           self.device_config,
                           lora_config=self.lora_config,
                           parallel_config=self.parallel_config,
                           scheduler_config=self.scheduler_config)

    self.model = _ipex_optimize_model(self.model, [], ggml_tensor_qtype["sym_int4"])


from ipex_llm.transformers.convert_ipex import (
    _ipex_optimize_rmsnorm, _ipex_optimize_attention, _ipex_optimize_decoder
)
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _using_tpp,
    _disable_tpp
)


def _ipex_optimize_model(model, rms_classes, qtype):
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.transformers.optimize import ipex_quantization_flow

    _disable_tpp()
    if qtype == ggml_tensor_qtype["bf16"]:
        model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True).eval()
    elif qtype == ggml_tensor_qtype["sym_int4"]:
        act_quant_mode_dict = {
            "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
            "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
            "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
            "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
        }
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=torch.quint4x2,  # INT4
            lowp_mode=ipex.quantization.WoqLowpMode.INT8,
            act_quant_mode=act_quant_mode_dict["PER_TENSOR"],
            group_size=-1,
        )
        model = ipex_quantization_flow(model, torch.bfloat16, None, qconfig, None)
    elif qtype == ggml_tensor_qtype["sym_int8"]:
        act_quant_mode_dict = {
            "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
            "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
            "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
            "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
        }
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=torch.qint8,  # INT8
            lowp_mode=ipex.quantization.WoqLowpMode.INT8,
            act_quant_mode=act_quant_mode_dict["PER_IC_BLOCK"],
            group_size=-1,
        )
        model = ipex_quantization_flow(model, torch.bfloat16, None, qconfig, None)
    return model


from intel_extension_for_pytorch.quantization._quantize import (
    may_quantize_deepspeed_modules, IPEX_WEIGHT_ONLY_QUANTIZATION_MODULE_CPU,
    DynamicQuantizedLinearLayer, DynamicQuantizedLinearAllreduce,
    DynamicQuantizedLmHeadLinearAllreduce,
    _may_insert_deepspeed_modules
)
from intel_extension_for_pytorch.quantization._quantize_utils import (
    auto_convert, copy_prepared_model
)


@functools.lru_cache(None)
def IPEX_DYNAMIC_QUANTIZATION_MODULE_CPU():
    torch_modules = {
        torch.nn.Linear: DynamicQuantizedLinearLayer,
        ColumnParallelLinear: DynamicQuantizedLinearLayer,
        RowParallelLinear: DynamicQuantizedLinearLayer,
        QKVParallelLinear: DynamicQuantizedLinearLayer,
        MergedColumnParallelLinear: DynamicQuantizedLinearLayer,
    }

    torch_modules = _may_insert_deepspeed_modules(
        torch_modules,
        DynamicQuantizedLinearLayer,
        DynamicQuantizedLinearAllreduce,
        DynamicQuantizedLmHeadLinearAllreduce,
    )
    return torch_modules


def _ipex_quantize_convert(model, inplace=False):
    r"""
    Convert an FP32 prepared model to a model which will automatically insert fake quant
    before a quantizable module or operator.

    Args:
        model (torch.nn.Module): The FP32 model to be convert.
        inplace: (bool): It will change the given model in-place if True.

    Returns:
        torch.nn.Module
    """
    assert isinstance(  # noqa
        model, torch.nn.Module
    ), "Only support nn.Module convert for quantization path"
    assert hasattr(  # noqa
        model, "q_config"
    ), "Please do prepare the model before doing convert"

    if inplace:
        convert_model = model
    else:
        try:
            convert_model = copy_prepared_model(model)
        except BaseException:
            AssertionError(
                False
            ), "The model's copy is failed, please try set inplace to True to do the convert"

    # For weight only quantization. Activation's observer is also PlaceholderObserver.
    if (
        isinstance(convert_model.q_config.activation(), PlaceholderObserver)
        and not convert_model.q_config.activation().is_dynamic
    ):
        qconfig_spec = {
            torch.nn.Linear: convert_model.q_config,
            torch.nn.LSTM: convert_model.q_config,
            torch.nn.GRU: convert_model.q_config,
            torch.nn.LSTMCell: convert_model.q_config,
            torch.nn.RNNCell: convert_model.q_config,
            torch.nn.GRUCell: convert_model.q_config,
            ColumnParallelLinear: convert_model.q_config,
            RowParallelLinear: convert_model.q_config,
            QKVParallelLinear: convert_model.q_config,
            MergedColumnParallelLinear: convert_model.q_config,
        }
        module_mappings = get_default_dynamic_quant_module_mappings().copy()
        module_mappings[
            torch.nn.Linear
        ] = nn.modules.weight_only_quantization.IpexWoqLinear
        module_mappings[ColumnParallelLinear] = nn.modules.weight_only_quantization.IpexWoqLinear
        module_mappings[RowParallelLinear] = nn.modules.weight_only_quantization.IpexWoqLinear
        module_mappings[QKVParallelLinear] = nn.modules.weight_only_quantization.IpexWoqLinear
        module_mappings[
            MergedColumnParallelLinear
        ] = nn.modules.weight_only_quantization.IpexWoqLinear

        module_mappings, qconfig_spec = may_quantize_deepspeed_modules(
            IPEX_WEIGHT_ONLY_QUANTIZATION_MODULE_CPU(),
            convert_model.q_config,
            module_mappings,
            qconfig_spec,
        )
        converted_model = torch.quantization.quantize_dynamic(
            convert_model,
            qconfig_spec=qconfig_spec,
            dtype=torch.qint8,
            mapping=module_mappings,
            inplace=inplace,
        )
        return converted_model

    # If the module's activation's qconfig is PlaceholderObserver,
    # we can say that the module want to run dynamic quantization path.
    if isinstance(convert_model.q_config.activation(), PlaceholderObserver):
        module_mappings = get_default_dynamic_quant_module_mappings()
        qconfig_spec = {
            torch.nn.Linear: convert_model.q_config,
            torch.nn.LSTM: convert_model.q_config,
            torch.nn.GRU: convert_model.q_config,
            torch.nn.LSTMCell: convert_model.q_config,
            torch.nn.RNNCell: convert_model.q_config,
            torch.nn.GRUCell: convert_model.q_config,
            ColumnParallelLinear: convert_model.q_config,
            RowParallelLinear: convert_model.q_config,
            QKVParallelLinear: convert_model.q_config,
            MergedColumnParallelLinear: convert_model.q_config,
        }

        module_mappings, qconfig_spec = may_quantize_deepspeed_modules(
            IPEX_DYNAMIC_QUANTIZATION_MODULE_CPU(),
            convert_model.q_config,
            module_mappings,
            qconfig_spec,
        )

        return torch.quantization.quantize_dynamic(
            convert_model,
            qconfig_spec=qconfig_spec,
            mapping=module_mappings,
            inplace=True,
        )

    # Convert linear, conv, and Embedding's weight dtype when use autocast,
    # which will reduce the dtype conversion.
    # TODO: check whether can be removed or not?
    if (
        torch.is_autocast_cpu_enabled()
        and torch.get_autocast_cpu_dtype() == torch.bfloat16
    ):
        convert_model = nn.utils._model_convert.convert_model_data_type(
            convert_model, torch.bfloat16
        )[1]

    convert_model = auto_convert(convert_model)
    return convert_model


from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _all_reduce_and_bias_add,
    _pre_ipex_gemm,
)
from intel_extension_for_pytorch.quantization import (
    QConfigWoq,
    quantize_per_channel,
    quantize_per_block,
)


@classmethod
def _ipex_woq_from_float(cls, mod, scales=None, zero_points=None):
    r"""Create a weight-only quantized module from a float module or qparams_dict

    Args:
        mod (Module): an instance of nn.Linear or its subclasses.
        scales: the scales Tensor for quantizing weight. If it is None,
            scales are found by min/max of the weight.
        zero_points: the zero points Tensor for quantizing weight. If it is None,
            zero points are found by min/max of the weight.
    """
    float_modules = [torch.nn.Linear, ColumnParallelLinear, RowParallelLinear,
                     QKVParallelLinear, MergedColumnParallelLinear]
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        float_modules.extend(deepspeed_modules)
    if any(issubclass(type(mod), float_module) for float_module in float_modules):
        float_modules.extend([type(mod)])

    assert type(mod) in float_modules, (  # noqa
        "IpexWoqLinear.from_float only works for one of"
        + str([float_mod.__name__ for float_mod in float_modules])
        + f" or their subclasses, but found {type(mod)}"
    )
    assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"  # noqa
    qconfig = mod.qconfig
    if qconfig is None or not isinstance(qconfig, QConfigWoq):
        return mod

    lowp_mode = qconfig.lowp_mode
    if qconfig.lowp_mode == 3 and qconfig.weight_dtype != torch.quint4x2:
        # lowp_mode=3 (INT8) is enabled for INT4 weight only
        # Fall back to lowp_mode=2 in other case
        # TODO(Weiwen) Support lowp_mode=3
        lowp_mode = 2
        print(
            "Warning: lowp_mode=3(INT8) is not supported yet in this case. "
            "Falling back to 2(BF16)."
        )
    act_quant_mode = qconfig.act_quant_mode
    num_concats = 1
    if hasattr(mod, "_num_concats"):
        num_concats = mod._num_concats
    dtype = qconfig.weight_dtype
    is_int4 = dtype == torch.quint4x2
    group_size = qconfig.group_size

    if group_size == -1:
        qweight, scales, zero_points = quantize_per_channel(
            mod.weight, is_int4, scales, zero_points
        )
    else:
        qweight, scales, zero_points = quantize_per_block(
            mod.weight, is_int4, group_size, scales, zero_points
        )
    if not hasattr(mod, "in_features"):
        mod.in_features = mod.weight.size()[1]
    if not hasattr(mod, "out_features"):
        mod.out_features = mod.weight.size()[0]

    qlinear = cls._init_cls(
        mod,
        dtype,
        qweight,
        scales,
        zero_points,
        group_size,
        lowp_mode,
        num_concats,
        act_quant_mode,
    )
    del qweight
    return qlinear


from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat
import torch.nn.functional as F
from intel_extension_for_pytorch.quantization._utils import _lstm_forward


def _ipex_module_call_to_function_call(module, args, weights):
    r"""
    This function is a help function which replace nn.module call to funtion call, which implement
    the nn.module's forward function.
    """
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d):
        output = module._conv_forward(args[0], weights[0], module.bias)
    elif (
        isinstance(module, torch.nn.Linear)
        or isinstance(module, ColumnParallelLinear)
        or isinstance(module, RowParallelLinear)
        or isinstance(module, QKVParallelLinear)
        or isinstance(module, MergedColumnParallelLinear)
    ):
        output = F.linear(args[0], weights[0], module.bias)
    elif isinstance(module, torch.nn.EmbeddingBag):
        output = F.embedding_bag(
            args[0],
            weights[0],
            args[1],
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.mode,
            module.sparse,
            args[2] if len(args) == 3 else None,
            module.include_last_offset,
            module.padding_idx,
        )
    elif isinstance(module, MergedEmbeddingBagWithCat):
        output = torch.ops.torch_ipex.merged_embeddingbag_cat_forward(
            weights, args[0], args[1], args[2]
        )
    elif isinstance(module, torch.nn.ConvTranspose2d) or isinstance(
        module, torch.nn.ConvTranspose3d
    ):
        if module.padding_mode != "zeros":
            raise ValueError(  # noqa
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )
        assert isinstance(module.padding, tuple)  # noqa
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_size = args[1] if len(args) == 2 else None
        # master code
        """
        num_spatial_dims = 2 if isinstance(module, torch.nn.ConvTranspose2d) else 3
        output_padding = module._output_padding(args[0], output_size,
                        module.stride, module.padding, module.kernel_size,
                        num_spatial_dims, module.dilation)
        """
        output_padding = module._output_padding(
            args[0],
            output_size,
            module.stride,
            module.padding,
            module.kernel_size,
            module.dilation,
        )
        # output_padding = module._output_padding(*arg_to)
        if isinstance(module, torch.nn.ConvTranspose2d):
            output = F.conv_transpose2d(
                args[0],
                weights[0],
                module.bias,
                module.stride,
                module.padding,
                output_padding,
                module.groups,
                module.dilation,
            )
        else:
            output = F.conv_transpose3d(
                args[0],
                weights[0],
                module.bias,
                module.stride,
                module.padding,
                output_padding,
                module.groups,
                module.dilation,
            )
    elif isinstance(module, torch.nn.LSTM):
        output = _lstm_forward(
            module, args[0], args[1] if len(args) == 2 else None, weights
        )
    return output


def _ipex_rmsnorm_forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        ops.fused_add_rms_norm(
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
        )
        return x, residual
    out = torch.ops.torch_ipex.rmsnorm(
        x,
        self.weight.data,
        self.variance_epsilon,
    )
    return out
