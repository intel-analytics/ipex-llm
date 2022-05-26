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
from bigdl.nano.utils.log4Error import invalidInputError


def load_inc_model(path, model, framework):
    if framework == 'pytorch':
        from .pytorch.quantized_model import PytorchQuantizedModel
        return PytorchQuantizedModel._load(path, model)
    elif framework == 'tensorflow':
        invalidInputError(False, "QuantizedTensorflowModel loading is not implemented yet.")
    else:
        invalidInputError(False,
                          "The value {} for framework is not supported."
                          " Please choose from 'pytorch'/'tensorflow'.")


def quantize(model, dataloader=None, metric=None, **kwargs):
    if kwargs['approach'] not in ['static', 'dynamic']:
        invalidInputError(False,
                          "Approach should be 'static' or 'dynamic', "
                          "{} is invalid.".format(kwargs['approach']))
    not_none_kwargs = {}
    for k, v in kwargs.items():
        # pop None values to use default
        if v is not None:
            not_none_kwargs[k] = v
    approach_map = {
        'static': 'post_training_static_quant',
        'dynamic': 'post_training_dynamic_quant'
    }
    not_none_kwargs['approach'] = approach_map.get(kwargs['approach'], None)
    quantizer = None
    if 'pytorch' in not_none_kwargs['framework']:
        from .pytorch.quantization import PytorchQuantization
        quantizer = PytorchQuantization(**not_none_kwargs)
    if 'onnx' in not_none_kwargs['framework']:
        invalidInputError('torch' in str(type(dataloader)),
                          errMsg="ONNXRuntime quantization only support in Pytorch.")
        from .onnx.pytorch.quantization import PytorchONNXRuntimeQuantization
        quantizer = PytorchONNXRuntimeQuantization(**not_none_kwargs)
    if 'tensorflow' in not_none_kwargs['framework']:
        from .tensorflow.quantization import TensorflowQuantization
        quantizer = TensorflowQuantization(**not_none_kwargs)
    if not quantizer:
        # default quantization
        from .core import BaseQuantization
        quantizer = BaseQuantization(**not_none_kwargs)
    return quantizer.post_training_quantize(model, dataloader, metric)
