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

import torch
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import List, Union, Any
from intel_npu_acceleration_library.backend.runtime import set_contiguous, record_function
from intel_npu_acceleration_library.backend.runtime import adapt_output_tensor, _model_cache
from collections import deque
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from ipex_llm.utils.common import invalidInputError
from transformers.utils import logging
from filelock import FileLock
import ctypes

logger = logging.get_logger(__name__)


@torch.no_grad()
def run_model(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: str,
    replica: int = 1,
) -> torch.Tensor:
    global _model_cache
    import time

    t0 = time.perf_counter()

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    if not isinstance(x, (list, tuple)):
        x = [x]

    # Reshape input
    input_dtype = x[0].dtype
    x_np = [set_contiguous(elem).to(torch.float16).numpy() for elem in x]
    op_args = []
    op_args_flatten = []
    for w in weights:
        if isinstance(w, tuple):  # from QuantizedLinear
            op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy()))
            op_args_flatten.append(op_args[-1][0])
            op_args_flatten.append(op_args[-1][1])
        else:
            op_args.append(set_contiguous(w).to(torch.float16).numpy())
            op_args_flatten.append(op_args[-1])

    shape_dtype_signature = "_".join(
        ["_".join(str(dim) for dim in t.shape) + f"_{t.dtype}" for t in x_np + op_args_flatten]
    )
    key = f"{backend_cls.func.__name__}_{shape_dtype_signature}"
    models = _model_cache.get(key, None)

    input_shapes = [elem.shape for elem in x_np]
    if models is None:
        _model_cache[key] = deque([backend_cls(*input_shapes) for i in range(replica)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(*input_shapes))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    with record_function(f"npu_factory_mul_{key}"):
        ret = model.run(x_np, *op_args, **op_kwargs)

    if isinstance(ret, list):
        results = [adapt_output_tensor(r, r.shape, input_dtype) for r in ret]
    else:
        results = adapt_output_tensor(ret, ret.shape, input_dtype)

    return results


class LLMBaseNNFactory(NNFactory):

    def __init__(self, max_seq_len, transpose_value, profile=False, device="NPU"):
        super().__init__(profile, device)
        self.cache_parameter_ops = []
        self.input_ops = []
        self.linear_ops = []
        self.kv_cache_c_handle = None
        self.kv_cache_torch = []
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value

    def create_cache_op(self, shape):
        invalidInputError(len(self.linear_ops) == 0,
                          "create_cache_op should be called before any linear op")
        op = super().parameter(shape)
        self.cache_parameter_ops.append(op)
        return op

    def create_input_op(self, shape):
        invalidInputError(len(self.cache_parameter_ops) == 0,
                          "create_input_op should be called before any create_cache_op")
        invalidInputError(len(self.linear_ops) == 0,
                          "create_input_op should be called before any linear op")

        op = super().parameter(shape)
        self.input_ops.append(op)
        return op

    def linear(self, *args, **kwargs):
        op = super().linear(*args, **kwargs)
        self.linear_ops.append(op)
        return op

    def parameter(self, shape):
        invalidInputError(False,
                          ("parameter should not be called directly, "
                           "use create_cache_op or create_input_op instead"))

    def update_cache(self, past_key_value, indexes):

        if self.kv_cache_c_handle is not None:
            curr_ptr = self.kv_cache_torch[0].storage().data_ptr()
            new_ptr = past_key_value.key_cache[indexes[0]].storage().data_ptr()
            if curr_ptr != new_ptr:
                backend_lib.destroyParameters(self.kv_cache_c_handle)
                self.kv_cache_c_handle = None
                self.kv_cache_torch = []
        if self.kv_cache_c_handle is None:
            for idx in indexes:
                past_key = past_key_value.key_cache[idx]
                past_value = past_key_value.value_cache[idx]
                invalidInputError(
                    past_key.dtype == torch.float16, f"past_key dtype is {past_key.dtype}"
                )
                new_size = (past_key.size(0), past_key.size(1), self.max_seq_len, past_key.size(3))
                past_key = past_key.as_strided(new_size, past_key.stride(), storage_offset=0)
                invalidInputError(past_key.is_contiguous(), "past_key is not contiguous")
                past_value = past_value.as_strided(new_size, past_value.stride(), storage_offset=0)
                if self.transpose_value:
                    past_value = past_value.transpose(-1, -2)
                invalidInputError(past_value.is_contiguous(), "past_value is not contiguous")

                self.kv_cache_torch.append(past_key)
                self.kv_cache_torch.append(past_value)

            layer_kv_cache_np = [p.numpy() for p in self.kv_cache_torch]
            invalidInputError(len(self.cache_parameter_ops) == len(layer_kv_cache_np),
                              (f"kv_cache size does not match graph, "
                               f"with kv_cache size: {len(layer_kv_cache_np)} and"
                               f" graph size: {len(self.cache_parameter_ops)}")
                              )
            self.kv_cache_c_handle = self.create_parameters(layer_kv_cache_np)
            self.load_cache_async()

    def load_cache_async(self):
        self.load_wt_fn(len(self.input_ops), self._mm, self.kv_cache_c_handle)

    def set_weights(self, op_id, weights):
        self.set_weights_async(op_id, weights)
        with FileLock(f"decoder_run.lock"):
            backend_lib.run(self._mm)

    def set_weights_async(self, op_id, weights):
        offset = len(self.input_ops) + len(self.cache_parameter_ops)
        invalidInputError(len(weights) == len(self.linear_ops),
                          (f"weights size does not match graph, "
                           f"with weights size: {len(weights)} and "
                           f" graph linear size: {len(self.linear_ops)}"))
        self.setWeights(offset, op_id, *weights)

    @staticmethod
    def run_decoders(inputs, decoders):
        x_np = [elem.to(torch.float16).numpy() for elem in inputs]

        num_decoders = len(decoders)
        num_inputs = len(x_np)

        with record_function(f"npu_factory"):

            array_type = ctypes.POINTER(ctypes.c_char) * num_decoders
            models_ptr = array_type(
                *[decoders[i]._mm for i in range(num_decoders)]
            )
            inputs_ptr = (ctypes.c_void_p * num_inputs)(
                *[x.ctypes.data_as(ctypes.c_void_p) for x in x_np]
            )
            backend_lib.run_decoders(models_ptr, inputs_ptr, num_decoders, num_inputs)

        hidden_states = decoders[-1].torch_out[0]
        new_key_states = []
        new_value_states = []
        for i in range(num_decoders):
            for j in range(1, len(decoders[i].torch_out)):
                if j % 2 == 1:
                    new_key_states.append(decoders[i].torch_out[j])
                else:
                    new_value_states.append(decoders[i].torch_out[j])
        return hidden_states, new_key_states, new_value_states
