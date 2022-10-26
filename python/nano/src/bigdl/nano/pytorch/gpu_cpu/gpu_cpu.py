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
from functools import partial
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10


CPU_DEVICE = torch.device("cpu")
TORCH_CUDA_NO_OP_LIST = ["set_device", "synchronize", "reset_peak_memory_stats",
                         "reset_accumulated_memory_stats"]
COMMON_TENSOR_TYPE = ['Double', 'Float', 'Long', 'Int', 'Short',
                      'Char', 'Byte', 'Half', 'Bool', 'BFloat16']
CUDA_TENSOR_TYPE = ['ComplexDouble', 'ComplexFloat']
CREATE_TENSOR_FUNC = ['rand', 'randint', 'randn', 'zeros', 'ones', 'empty', 'full',
                      'rand_like', 'randint_like', 'randn_like', 'zeros_like',
                      'ones_like', 'empty_like', 'full_like',
                      'tensor', 'scalar_tensor', 'sparse_coo_tensor', 'sparse_csr_tensor',
                      'randperm', 'normal', 'range', 'arange', 'eye',
                      'as_tensor', 'asarray',
                      'linspace', 'logspace',
                      'tril_indices', 'triu_indices',
                      'bartlett_window', 'blackman_window', 'hamming_window',
                      'hann_window', 'kaiser_window',
                      'empty_quantized', 'empty_strided',
                      'frombuffer', 'from_file']


def np_op_func(*args, **kwargs):
    pass


def cuda(self, *args, **kwargs):
    return self


def is_gpu_device(device):
    return isinstance(device, int) or \
        (isinstance(device, str) and 'cuda' in device) or \
        (isinstance(device, torch.device) and device.type == 'cuda')


def to(torch_to):
    def new_to(self, *args, **kwargs):
        if 'device' in kwargs and is_gpu_device(kwargs['device']):
            kwargs['device'] = 'cpu'
            return torch_to(self, *args, **kwargs)
        elif len(args) > 0 and is_gpu_device(args[0]):
            return torch_to(self, 'cpu', *args[1:], **kwargs)
        else:
            return torch_to(self, *args, **kwargs)
    return new_to


def load(torch_load):
    def new_load(*args, **kwargs):
        if 'map_location' in kwargs:
            kwargs['map_location'] = 'cpu'
            return torch_load(*args, **kwargs)
        elif len(args) > 1:
            return torch_load(args[0], 'cpu', *args[2:], **kwargs)
        else:
            return torch_load(*args, **kwargs)
    return new_load


class DeviceClass:
    def __new__(cls, string):
        return CPU_DEVICE


def GradScalerClass_wrapper(GradScaler):
    class GradScalerClass:
        def __new__(cls, *args, **kwargs):
            kwargs["enabled"] = False
            return GradScaler(*args, **kwargs)
    return GradScalerClass


class no_op_context:
    def __init__(self, *args, **kargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def wait_stream(self, *args, **kargs):
        pass


def current_stream():
    return no_op_context()


def init_process_group(torch_init_process_group):
    def new_init_process_group(backend, *args, **kargs):
        if backend == 'nccl':
            torch_init_process_group('gloo', *args, **kargs)
        else:
            torch_init_process_group(backend, *args, **kargs)
    return new_init_process_group


def create_tensor_func(torch_create_tensor_func):
    def new_create_tensor_func(*args, **kwargs):
        if 'device' in kwargs and is_gpu_device(kwargs['device']):
            kwargs['device'] = 'cpu'
        return torch_create_tensor_func(*args, **kwargs)
    return new_create_tensor_func


def patch_cuda(disable_jit=True):
    # add this parameter since it's a known issue
    if disable_jit:
        torch.jit._state.disable()

    setattr(torch.Tensor, "cuda", cuda)
    setattr(torch.Tensor, "to", to(torch.Tensor.to))
    setattr(torch.nn.Module, "cuda", cuda)
    setattr(torch.nn.Module, "to", to(torch.nn.Module.to))
    setattr(torch, "device", DeviceClass)
    setattr(torch, "load", load(torch.load))
    setattr(torch.cuda, "Stream", no_op_context)
    setattr(torch.cuda, "current_stream", current_stream)
    setattr(torch.Tensor, "record_stream", np_op_func)
    if not TORCH_VERSION_LESS_1_10:
        setattr(torch.cuda.amp, "autocast", torch.cpu.amp.autocast)
    setattr(torch.cuda.amp, "GradScaler", GradScalerClass_wrapper(torch.cuda.amp.GradScaler))
    setattr(torch.distributed, "init_process_group",
            init_process_group(torch.distributed.init_process_group))
    for no_op_cand in TORCH_CUDA_NO_OP_LIST:
        setattr(torch.cuda, no_op_cand, np_op_func)
    for t in COMMON_TENSOR_TYPE:
        setattr(torch.cuda, f'{t}Tensor', getattr(torch, f'{t}Tensor'))
    for t in CUDA_TENSOR_TYPE:
        setattr(torch.cuda, f'{t}Tensor', getattr(torch, f'{t.replace("Complex", "")}Tensor'))
    for f in CREATE_TENSOR_FUNC:
        try:
            setattr(torch, f, create_tensor_func(getattr(torch, f)))
        except AttributeError:
            pass
