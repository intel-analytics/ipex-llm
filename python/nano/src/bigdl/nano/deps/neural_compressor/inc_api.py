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


import operator
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import compare_version


def load_inc_model(path, model, framework, input_sample=None):
    if framework == 'pytorch':
        from .pytorch.quantized_model import PytorchQuantizedModel
        # only ipex quantization needs example_inputs
        return PytorchQuantizedModel._load(path, model, example_inputs=input_sample)
    elif framework == 'tensorflow':
        from .tensorflow.model import KerasQuantizedModel
        return KerasQuantizedModel._load(path, model)
    else:
        invalidInputError(False,
                          "The value {} for framework is not supported."
                          " Please choose from 'pytorch'/'tensorflow'.")


def quantize(model, dataloader=None, eval_func=None, metric=None,
             thread_num=None, **kwargs):
    if compare_version("neural_compressor", operator.ge, "2.0"):
        from .inc_api_2 import quantize
        return quantize(model, dataloader, eval_func, metric, thread_num, **kwargs)

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
    onnxruntime_session_options = not_none_kwargs.pop('onnxruntime_session_options', None)
    if 'pytorch' in not_none_kwargs['framework']:
        from .pytorch.quantization import PytorchQuantization
        quantizer = PytorchQuantization(thread_num=thread_num, **not_none_kwargs)
    if 'onnx' in not_none_kwargs['framework']:
        onnx_option = not_none_kwargs.pop('onnx_option', None)
        if onnxruntime_session_options is None:
            import onnxruntime
            onnxruntime_session_options = onnxruntime.SessionOptions()
        if thread_num is not None:
            onnxruntime_session_options.intra_op_num_threads = thread_num
            onnxruntime_session_options.inter_op_num_threads = thread_num
        if onnx_option == 'tensorflow':
            from .onnx.tensorflow.quantization import KerasONNXRuntimeQuantization
            quantizer = KerasONNXRuntimeQuantization(
                onnxruntime_session_options=onnxruntime_session_options, **not_none_kwargs)
        else:
            from .onnx.pytorch.quantization import PytorchONNXRuntimeQuantization
            quantizer = PytorchONNXRuntimeQuantization(
                onnxruntime_session_options=onnxruntime_session_options, **not_none_kwargs)
    if 'tensorflow' in not_none_kwargs['framework']:
        from .tensorflow.quantization import TensorflowQuantization
        quantizer = TensorflowQuantization(**not_none_kwargs)
    if not quantizer:
        # default quantization
        from .core import BaseQuantization
        quantizer = BaseQuantization(**not_none_kwargs)
    return quantizer.post_training_quantize(model, dataloader, metric)
