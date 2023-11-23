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


import copy
import yaml
from pathlib import Path

import torch
import torch.nn as nn

from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object, \
    transform_state_dict_to_dtype


def load_model(path, model: nn.Module = None, input_sample=None,
               inplace=False, device=None, cache_dir=None, shapes=None):
    """
    Load a model from local.

    :param path: Path to model to be loaded. Path should be a directory.
    :param model: Required FP32 model to load pytorch model, it is needed if:
               1. you accelerated the model with accelerator=None by
               InferenceOptimizer.trace/InferenceOptimizer.quantize.
               2. you want to the loaded model contains the attributes of original model.
    :param input_sample: Input sample for your model, could be a Tensor or a tuple.
               This parameter is needed if:
               1. saving model is accelerated by INC IPEX quantization.
               2. saving model is accelerated by JIT and you set compression='bf16'
               when saving.
    :param inplace: whether to perform inplace optimization. Default: ``False``.
    :param device: A string represents the device of the inference. Default to None.
                   Only valid for openvino model, otherwise will be ignored.
    :param cache_dir: A directory for OpenVINO to cache the model. Default to None.
                      Only valid for openvino model, otherwise will be ignored.
    :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',
               '[1,3,224,224]'. This parameter affect model Parameter shape, can be
               dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.
               Only valid for openvino model, otherwise will be ignored.
    :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or
                precision(FP32/FP16/BF16/INT8).
    """
    from bigdl.nano.pytorch.amp.amp_api import load_bf16_model
    from bigdl.nano.pytorch.low_precision.jit_int8_api import load_pytorchjitint8_model
    from bigdl.nano.pytorch.context_manager import generate_context_manager
    from bigdl.nano.deps.openvino.openvino_api import load_openvino_model
    from bigdl.nano.deps.ipex.ipex_api import load_ipexjit_model, load_ipexjitbf16_model,\
        load_ipex_quantization_model, load_ipex_xpu_model
    from bigdl.nano.deps.onnxruntime.onnxruntime_api import load_onnxruntime_model
    from bigdl.nano.deps.neural_compressor.inc_api import load_inc_model

    if isinstance(path, dict):
        metadata = yaml.safe_load(path["nano_model_meta.yml"])
        path["nano_model_meta.yml"].seek(0)
    else:
        path = Path(path)
        if not path.exists():
            invalidInputError(False, "{} doesn't exist.".format(path))
        meta_path = path / "nano_model_meta.yml"
        if not meta_path.exists():
            invalidInputError(False, "File {} is required to load model.".format(str(meta_path)))
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
    model_type = metadata.get('ModelType', None)
    result = None
    if model_type == 'PytorchOpenVINOModel':
        result = load_openvino_model(path, device=device, cache_dir=cache_dir, shapes=shapes)
    if model_type == 'PytorchONNXRuntimeModel':
        result = load_onnxruntime_model(path)
    if model_type == 'PytorchQuantizedModel':
        result = load_inc_model(path, model, 'pytorch', input_sample=input_sample)
    if model_type == 'PytorchIPEXJITModel':
        result = load_ipexjit_model(path, model, inplace=inplace,
                                    input_sample=input_sample)
    if model_type == 'PytorchIPEXJITBF16Model':
        result = load_ipexjitbf16_model(path, model, inplace=inplace,
                                        input_sample=input_sample)
    if model_type == 'PytorchIPEXQuantizationModel':
        result = load_ipex_quantization_model(path, model, inplace=inplace)
    if model_type == 'BF16Model':
        return load_bf16_model(path, model)
    if model_type == 'PytorchJITINT8Model':
        return load_pytorchjitint8_model(path)
    if model_type == 'PytorchIPEXPUModel':
        return load_ipex_xpu_model(path, model)
    if result is not None:
        if isinstance(model, torch.nn.Module):
            # patch attributes to accelerated model
            patch_attrs_from_model_to_object(model, result)
        return result
    if isinstance(model, nn.Module):
        # typically for models of nn.Module, pl.LightningModule type
        model = copy.deepcopy(model)
        checkpoint_path = metadata.get('checkpoint', None)
        thread_num = None
        if "thread_num" in metadata and metadata["thread_num"] is not None:
            thread_num = int(metadata["thread_num"])
        if checkpoint_path:
            if isinstance(path, dict):
                checkpoint_path = path[metadata['checkpoint']]
            else:
                checkpoint_path = path / metadata['checkpoint']
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if metadata['compression'] == "bf16":
                state_dict = transform_state_dict_to_dtype(state_dict, dtype="fp32")
            model.load_state_dict(state_dict)
            # patch ContextMagager to original model to keep behaviour consitent
            model._nano_context_manager = \
                generate_context_manager(accelerator=None,
                                         precision="fp32",
                                         thread_num=thread_num)  # type: ignore
            return model
        else:
            invalidInputError(False, "Key 'checkpoint' must be specified.")
    else:
        invalidInputError(False,
                          "ModelType {} or argument 'model={}' is not acceptable for pytorch"
                          " loading.".format(model_type, type(model)))
