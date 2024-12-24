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
# Some parts of this file is adapted from
# https://github.com/intel/auto-round/blob/main/auto_round/auto_quantizer.py
# and
# https://github.com/intel/auto-round/blob/main/auto_round/backend.py
# which is licensed under Apache License 2.0:
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import importlib
import torch.nn as nn
from transformers.utils.versions import require_version
from transformers.pytorch_utils import Conv1D
from logging import getLogger
from typing import Union
from ipex_llm.utils.common import invalidInputError

logger = getLogger(__name__)

import auto_round


def check_compatible(backend_name, device, bits, group_size, sym, packing_format,
                     in_features, out_features, check_requirements=True):
    """Checks if the given configuration is compatible with the specified backend.

    Args:
        backend_name (str): The name of the backend to check compatibility for.
        device (str): The device on which the backend operates (e.g., 'cuda', 'cpu').
        bits (int): The bit-width of the quantization (e.g., 2, 4, 8).
        group_size (Optional[int]): The size of the quantization group. Can be None if
            not required by the backend.
        sym (bool): Whether symmetric quantization is required (True for symmetric).
        packing_format (str): The packing format used by the backend (e.g., 'triton').
        in_features (int): The number of input features for the model layer.
        out_features (int): The number of output features for the model layer.
        check_requirements (bool): Whether check the requirement

    Returns:
        bool: True if the configuration is compatible with the backend, False otherwise.

    Raises:
        KeyError: If the backend_name is not found in BackendInfos.

    Compatibility checks:
    - Device must match one of the backend's supported devices.
    - Bit-width must be supported by the backend.
    - If group_size is required by the backend, it must match.
    - Symmetric or asymmetric quantization must be supported.
    - If the packing format matches exactly, all feature checks must pass.
    - If the packing format does not match, it must be convertible.
    """
    backend = auto_round.backend.BackendInfos[backend_name]

    # Check if device is supported by the backend
    if device not in backend.device:
        return False

    # Check if bit-width is supported
    if bits not in backend.bits:
        return False

    # Check if group_size is valid (if required by backend)
    if backend.group_size is not None and group_size not in backend.group_size:
        return False

    # Check if symmetric/asymmetric quantization is supported
    if sym not in backend.sym:
        return False

    # Check packing format and apply feature checks
    if packing_format == backend.packing_format:
        for check in backend.feature_checks:
            if not check(in_features, out_features):
                return False

    # Check if the format is convertible when packing formats differ
    if packing_format != backend.packing_format and \
            packing_format not in backend.convertable_format:
        return False

    if check_requirements and backend.requirements is not None:
        for requirement in backend.requirements:
            if isinstance(requirement, str):
                try:
                    require_version(requirement)
                except ImportError:
                    return False
            else:
                res, _ = requirement()
                return res

    return True


def get_layer_backend(device, backend, orig_backend, bits, group_size, sym,
                      in_features, out_features):
    """Selects the most suitable backend for the layer based on compatibility and priority.

    This function first checks if the specified backend supports the layer with the
    provided configuration. If not, it iterates through other available backends,
    checking compatibility and returning the one with the highest priority.

    Args:
        device (str):
            The device on which the layer will run, e.g., 'cpu', 'cuda'.
        backend (str):
            The target backend to be used for this layer.
        orig_backend (str):
            The original backend from which packing format information is retrieved.
        bits (int):
            The number of bits used for quantization.
        group_size (int):
            The group size for quantization.
        sym (bool):
            Whether symmetric quantization is enabled.
        in_features (int):
            The number of input features for the layer.
        out_features (int):
            The number of output features for the layer.

    Returns:
        str:
            The selected backend that is compatible with the layer configuration.

    Raises:
        AssertionError:
            If the specified backend is not supported.
        ValueError:
            If no compatible backend is found for the given layer configuration.
    """
    # Check if the provided backend is in BackendInfos
    invalidInputError(backend in auto_round.backend.BackendInfos.keys(),
                      f"Unsupported backend {backend}, "
                      "please set it to `auto` to try automatic selection")

    packing_format = auto_round.backend.BackendInfos[orig_backend].packing_format

    # Check if the provided backend supports the layer configuration
    if check_compatible(backend, device, bits, group_size, sym, packing_format,
                        in_features, out_features):
        return backend

    # Find and store other compatible backends
    supported_backends = []
    for key in auto_round.backend.BackendInfos.keys():
        if key == backend:
            continue
        if check_compatible(key, device, bits, group_size, sym, packing_format,
                            in_features, out_features):
            supported_backends.append(key)

    # Raise an error if no compatible backends are found
    if len(supported_backends) == 0:
        supported_backends_need_package = []
        for key in auto_round.backend.BackendInfos.keys():
            if check_compatible(key, device, bits, group_size, sym, packing_format,
                                in_features, out_features,
                                check_requirements=False):
                supported_backends_need_package.append(key)

        if len(supported_backends_need_package) > 0:
            supported_backends_need_package = sorted(
                supported_backends_need_package,
                key=lambda support_backend:
                    auto_round.backend.BackendInfos[support_backend].priority,
                reverse=True)
            backend_info = auto_round.backend.BackendInfos[supported_backends_need_package[0]]
            # ipex-llm change start
            for requirement in backend_info.requirements:
                if isinstance(requirement, str) and \
                        not requirement.startswith("intel-extension-for-"):
                    try:
                        require_version(requirement)
                    except ImportError:
                        logger.error(f"pip install {requirement}")
                elif not requirement.startswith("intel-extension-for-"):
                    str_info = requirement()[1]
                    logger.error(str_info)
            if not requirement.startswith("intel-extension-for-"):
                invalidInputError(False,
                                  f"exit for missing requirement {requirement}")
            # ipex-llm change end

    # Sort the compatible backends by priority and return the one with the highest priority
    supported_backends = sorted(supported_backends,
                                key=lambda support_backend:
                                    auto_round.backend.BackendInfos[support_backend].priority,
                                reverse=True)

    # ipex-llm change start
    try:
        return supported_backends[0]
    except:
        return "ipex_gptq"
    # ipex-llm change end

import auto_round.backend
auto_round.backend.get_layer_backend = get_layer_backend
auto_round.backend.check_compatible = check_compatible

importlib.reload(auto_round.backend)

from auto_round.utils import (get_block_names, get_module, set_module,
                              get_multimodal_block_names, find_matching_blocks)


def cpu_post_init(self, model):
    return model


def convert_model(self, model: nn.Module):
    """Converts the given model to an AutoRound model by replacing its layers with quantized layers.

    This method extracts the quantization configuration from the model and adjusts its layers
    according to the specified quantization parameters. It supports different backends and
    ensures that the model's data type is compatible with the selected hardware.

    Args:
        model (nn.Module):
            The model to be converted into an AutoRound model.

    Returns:
        nn.Module:
            The converted AutoRound model with quantized layers.

    Raises:
        ValueError:
            If the quantization backend is not specified in the configuration.
    """

    from auto_round.utils import get_layer_names_in_block

    quantization_config = model.config.quantization_config
    if not hasattr(quantization_config, "target_backend"):
        quantization_config.target_backend = quantization_config.backend

    target_device = self.detect_device(quantization_config.target_backend,
                                       quantization_config.backend)
    self.target_device = target_device

    if hasattr(quantization_config, "backend"):  # pragma: no cover
        if ("hpu" == target_device or "cpu" == target_device) and model.dtype != torch.bfloat16:
            # ipex-llm code change start
            # model = model.to(torch.bfloat16)
            model = model.to(torch.float16)
            # ipex-llm code change end
        else:
            if model.dtype != torch.float16:
                model = model.to(torch.float16)

    bits = quantization_config.bits
    group_size = quantization_config.group_size
    data_type = quantization_config.data_type if hasattr(quantization_config,
                                                         "data_type") else "int"  # pragma: no cover
    sym = quantization_config.sym
    if hasattr(quantization_config, "to_quant_block_names"):
        to_quant_block_names = quantization_config.to_quant_block_names
    else:
        to_quant_block_names = None
    quant_block_list = quantization_config.quant_block_list if hasattr(quantization_config,
                                                                       "quant_block_list") else None
    if to_quant_block_names is None:  # TODO check compatibility
        all_blocks = get_block_names(model)
    else:
        all_blocks = get_multimodal_block_names(model, quant_vision=True)
    if quant_block_list is None:
        quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
    layer_names = get_layer_names_in_block(model, quant_block_list=quant_block_list)

    extra_config = {}
    if hasattr(quantization_config, "extra_config"):
        extra_config = quantization_config.extra_config

    layer_names += extra_config.keys()
    layer_names = list(set(layer_names))

    layer_configs = {}
    for layer_name in layer_names:
        layer_configs[layer_name] = {}
        if layer_name not in extra_config:
            layer_configs[layer_name]["bits"] = bits
            layer_configs[layer_name]["group_size"] = group_size
            layer_configs[layer_name]["data_type"] = data_type
            layer_configs[layer_name]["sym"] = sym
            layer_configs[layer_name]["clip"] = False
        else:
            layer_configs[layer_name]["bits"] = extra_config[layer_name].get("bits", bits)
            layer_configs[layer_name]["group_size"] = extra_config[layer_name].get("group_size",
                                                                                   group_size)
            layer_configs[layer_name]["data_type"] = extra_config[layer_name].get("data_type",
                                                                                  data_type)
            layer_configs[layer_name]["sym"] = extra_config[layer_name].get("sym", sym)
            layer_configs[layer_name]["clip"] = extra_config[layer_name].get("clip", False)

    if hasattr(quantization_config, "backend"):  # pragma: no cover
        backend = quantization_config.backend
    elif 'gptq' in quantization_config.quant_method:  # pragma: no cover
        backend = 'gptq'
    else:  # pragma: no cover
        invalidInputError(False, "Quantization backend must be specified.")

    self._replace_by_quant_layers(model, layer_configs, quantization_config.target_backend,
                                  target_device, backend)
    return model


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def _replace_by_quant_layers(self, module: nn.Module, layer_configs, target_backend,
                             target_device, orig_backend):
    """Replaces linear layers in the given module with quantized layers.

    This method iterates over the specified layer configurations and replaces
    the original layers in the module with instances of `QuantLinear`. It handles
    various layer types and ensures that the correct quantization parameters are applied.

    Args:
        module (nn.Module):
            The module containing layers to be quantized.
        layer_configs (dict):
            A dictionary containing configuration for each layer's quantization.
        target_backend (str):
            The backend to use for quantization, which includes device and format information.
        target_device (str):
            The device on which the model will run (e.g., 'cuda', 'cpu', 'hpu').
        orig_backend (str):
            The original backend of the packing.

    Raises:
        AssertionError:
            If any condition related to backend or quantization configuration is not met.
    """
    # ipex-llm code change start
    from auto_round.backend import dynamic_import_inference_linear
    # ipex-llm code change end

    def remove_device_str(s, device_str):
        if s and s.startswith(device_str):
            return s[len(device_str):].lstrip(":")
        return s

    if "auto" == target_backend.split(':')[0]:
        target_backend = target_backend[4:]  # Remove 'auto'
        if len(target_backend) >= 1 and target_backend[0] == ":":
            target_backend = target_backend[1:]

    # Remove device info from target_backend
    target_backend = remove_device_str(target_backend, "cpu")
    target_backend = remove_device_str(target_backend, "hpu")
    target_backend = remove_device_str(target_backend, "cuda")
    orig_backend = self.find_backend(orig_backend)

    if target_backend == "":
        target_backend = orig_backend

    self.need_marlin_repacking = False

    for layer_name in layer_configs.keys():
        config = layer_configs[layer_name]
        bits = config["bits"]
        group_size = config["group_size"]
        data_type = config["data_type"]
        sym = config["sym"]
        clip = config["clip"]

        if not (bits <= 8):
            continue

        layer = get_module(module, layer_name)
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
        elif isinstance(layer, nn.Conv2d):  # Not supported currently
            in_features = layer.in_channels
            out_features = layer.out_channels
        elif isinstance(layer, Conv1D):  # TODO: Needs verification
            in_features = layer.weight.shape[0]
            out_features = layer.weight.shape[1]
        else:
            continue

        if "marlin" in target_backend and "marlin" not in orig_backend:
            # Need to repack
            invalidInputError(sym,
                              "Marlin only supports symmetric quantization")
            invalidInputError(target_device == "cuda",
                              "Marlin only supports CUDA device")
            invalidInputError("awq" not in orig_backend,
                              "Marlin does not support repacking from AWQ format")
            self.need_marlin_repacking = True
            # Using original backend to load the layer then replace
            layer_backend = orig_backend
        else:
            target_backend = self.find_backend(target_backend)
            layer_backend = get_layer_backend(
                target_device, target_backend, orig_backend, bits, group_size,
                sym, in_features, out_features
            )
        if "gptq" in layer_backend and "exllamav2" in layer_backend:
            try:
                from exllamav2_kernels import gemm_half_q_half, make_q_matrix
            except:
                logger.warning_once(
                    "For better inference performance, please install exllamav2 kernel "
                    "via `pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@b8b4127`")

        QuantLinear = dynamic_import_inference_linear(layer_backend, bits, group_size, sym)

        layer_device = get_device(layer)

        bias = layer.bias is not None
        if "awq" in layer_backend:
            new_layer = QuantLinear.from_linear(  # pylint: disable=E1123
                layer,
                bits,
                group_size,
                init_only=True
            )
        else:
            try:
                new_layer = QuantLinear(  # pylint: disable=E1123
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=layer.weight.dtype,
                    clip=clip
                )
            except:
                new_layer = QuantLinear(  # pylint: disable=E1123
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=layer.weight.dtype,
                )

        new_layer.device = layer_device
        set_module(module, layer_name, new_layer)

auto_round.auto_quantizer.AutoRoundQuantizer.cpu_post_init = cpu_post_init
auto_round.auto_quantizer.AutoRoundQuantizer._replace_by_quant_layers = _replace_by_quant_layers
auto_round.auto_quantizer.AutoRoundQuantizer.convert_model = convert_model
