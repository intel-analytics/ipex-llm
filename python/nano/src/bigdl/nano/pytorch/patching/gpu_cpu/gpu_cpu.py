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
import torch
from logging import warning

from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_10


CPU_DEVICE = torch.device("cpu")
TORCH_CUDA_NO_OP_LIST = ["set_device", "synchronize", "reset_peak_memory_stats",
                         "reset_accumulated_memory_stats"]
COMMON_TENSOR_TYPE = ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte',
                      'Half', 'Bool', 'BFloat16']
CREATE_TENSOR_FUNC = ['rand', 'randint', 'randn', 'zeros', 'ones', 'empty', 'full',
                      'rand_like', 'randint_like', 'randn_like', 'zeros_like',
                      'ones_like', 'empty_like', 'full_like',
                      'tensor', 'scalar_tensor', 'sparse_coo_tensor', 'sparse_csr_tensor',
                      'sparse_csc_tensor', 'sparse_bsc_tensor', 'sparse_bsr_tensor',
                      'sparse_compressed_tensor', 'nested_tensor'
                      'randperm', 'normal', 'range', 'arange', 'eye',
                      'as_tensor', 'asarray',
                      'linspace', 'logspace',
                      'tril_indices', 'triu_indices',
                      'bartlett_window', 'blackman_window', 'hamming_window',
                      'hann_window', 'kaiser_window',
                      'empty_quantized', 'empty_strided',
                      'frombuffer', 'from_file']

attrs = []
is_cuda_patched = False


def replace_attr(obj, name: str, value):
    torch_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, torch_attr))


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
        if is_gpu_device(kwargs.get('device')):
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


# it seems we don't really need this repalcement
if not TORCH_VERSION_LESS_1_10:
    class new_autocast(torch.autocast):
        def __init__(self, device_type, dtype=None, *args, **kwargs):
            device_type = 'cpu' if device_type == 'cuda' else device_type
            dtype = torch.bfloat16 if dtype == torch.float16 else dtype
            super().__init__(device_type, dtype, *args, **kwargs)


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
        if is_gpu_device(kwargs.get('device')):
            kwargs['device'] = 'cpu'
        return torch_create_tensor_func(*args, **kwargs)
    return new_create_tensor_func


def patch_cuda(disable_jit: bool = True):
    '''
    patch_cuda is used to make users' application that is written for cuda only
    runnable on a CPU device by one-line patching.

    e.g.
        >>> from bigdl.nano.pytorch.patching import patch_cuda
        >>> patch_cuda()  # be sure it is used at the header of the application
        >>> # all other cuda only codes will be avilable for cpu

    :param disable_jit: bool, if to disable jit compile. This is a known issue
           for patch_cuda function. jit compile has not been supported for some
           of the patching. Users may change it to False to check if their application
           is affected by this issue.
    '''
    global is_cuda_patched
    if is_cuda_patched:
        return

    # add this parameter since it's a known issue
    if disable_jit:
        warning("This CUDA patch is incompatible with JIT, JIT will be disabled!")
        torch.jit._state.disable()

    replace_attr(torch.Tensor, "cuda", cuda)
    replace_attr(torch.Tensor, "to", to(torch.Tensor.to))
    replace_attr(torch.nn.Module, "cuda", cuda)
    replace_attr(torch.nn.Module, "to", to(torch.nn.Module.to))
    replace_attr(torch, "device", DeviceClass)
    replace_attr(torch, "load", load(torch.load))
    replace_attr(torch.cuda, "Stream", no_op_context)
    replace_attr(torch.cuda, "current_stream", current_stream)
    replace_attr(torch.Tensor, "record_stream", np_op_func)
    if not TORCH_VERSION_LESS_1_10:
        replace_attr(torch, "autocast", new_autocast)
        replace_attr(torch.cuda.amp, "autocast", torch.cpu.amp.autocast)
    replace_attr(torch.cuda.amp, "GradScaler", GradScalerClass_wrapper(torch.cuda.amp.GradScaler))
    replace_attr(torch.distributed, "init_process_group",
                 init_process_group(torch.distributed.init_process_group))
    for no_op_cand in TORCH_CUDA_NO_OP_LIST:
        replace_attr(torch.cuda, no_op_cand, np_op_func)
    for t in COMMON_TENSOR_TYPE:
        replace_attr(torch.cuda, f'{t}Tensor', getattr(torch, f'{t}Tensor'))
    for f in CREATE_TENSOR_FUNC:
        try:
            replace_attr(torch, f, create_tensor_func(getattr(torch, f)))
        except AttributeError:
            pass

    is_cuda_patched = True


def unpatch_cuda():
    '''
    unpatch_cuda is an reverse function to patch_cuda. It will change the application
    back to be available on cuda.

    e.g.
        >>> from bigdl.nano.pytorch.patching import unpatch_cuda
        >>> unpatch_cuda()  # be sure it is used after patch_cuda
        >>> # all other codes will be avilable for cuda

    :param disable_jit: bool, if to disable jit compile. This is a known issue
           for patch_cuda function. jit compile has not been supported for some
           of the patching. Users may change it to False to check if their application
           is affected by this issue.
    '''
    global is_cuda_patched
    if not is_cuda_patched:
        return

    torch.jit._state.enable()
    for obj, name, torch_attr in attrs:
        setattr(obj, name, torch_attr)

    is_cuda_patched = False


def get_cuda_status():
    return is_cuda_patched
