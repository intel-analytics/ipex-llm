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
import warnings
import inspect
from functools import wraps
import cpuinfo
import importlib
import pkg_resources
from pkg_resources import DistributionNotFound
from packaging.version import Version
from typing import Callable

from bigdl.nano.utils.log4Error import invalidInputError

from multiprocessing import Process
import multiprocessing as mp


def deprecated(func_name=None, message=""):
    def deprecated_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            funcname = function.__name__ if func_name is None else func_name
            warnings.warn("`{}` will be deprecated in future release. {}"
                          .format(funcname, message),
                          category=DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            return function(*args, **kwargs)
        return wrapped
    return deprecated_decorator


# code adapted from https://github.com/intel/neural-compressor/
#                   blob/master/neural_compressor/utils/utility.py#L203

def singleton(cls):
    instance = {}

    def _singleton(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return _singleton


@singleton
class CPUInfo():
    def __init__(self):
        self._bf16 = False
        self._avx512 = False
        self._check_extension_features()

    def _check_extension_features(self):
        info = cpuinfo.get_cpu_info()
        if 'arch' in info and 'X86' in info['arch']:
            # get cpu features from cpuid
            cpuid = cpuinfo.CPUID()
            max_extension_support = cpuid.get_max_extension_support()
            if max_extension_support >= 7:
                # get extended feature bits
                # EAX = 7, ECX = 0
                # https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
                ebx = cpuid._run_asm(
                    b"\x31\xC9",             # xor ecx, ecx
                    b"\xB8\x07\x00\x00\x00"  # mov eax, 7
                    b"\x0f\xa2"              # cpuid
                    b"\x89\xD8"              # mov ax, bx
                    b"\xC3"                  # ret
                )
                ecx = cpuid._run_asm(
                    b"\x31\xC9",             # xor ecx, ecx
                    b"\xB8\x07\x00\x00\x00"  # mov eax, 7
                    b"\x0f\xa2"              # cpuid
                    b"\x89\xC8"              # mov ax, cx
                    b"\xC3"                  # ret
                )
                edx = cpuid._run_asm(
                    b"\x31\xC9",             # xor ecx, ecx
                    b"\xB8\x07\x00\x00\x00"  # mov eax, 7
                    b"\x0f\xa2"              # cpuid
                    b"\x89\xD0"              # mov ax, dx
                    b"\xC3"                  # ret
                )
                avx512_f = bool(ebx & (1 << 16))
                avx512_vnni = bool(ecx & (1 << 11))
                amx_bf16 = bool(edx & (1 << 22))
                amx_tile = bool(edx & (1 << 24))

                # EAX = 7, ECX = 1
                # https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=1:_Extended_Features
                eax = cpuid._run_asm(
                    b"\xB9\x01\x00\x00\x00",  # mov ecx, 1
                    b"\xB8\x07\x00\x00\x00"   # mov eax, 7
                    b"\x0f\xa2"               # cpuid
                    b"\xC3"                   # ret
                )

                avx512_bf16 = bool(eax & (1 << 5))

                self._bf16 = avx512_bf16 or amx_bf16
                self._avx512 = avx512_f or avx512_vnni or amx_tile or self._bf16

    @property
    def has_bf16(self):
        return self._bf16

    @property
    def has_avx512(self):
        return self._avx512


class _append_return_to_pipe(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, queue, *args, **kwargs):
        res = self.func(*args, **kwargs)
        queue.put(res)


class spawn_new_process(object):
    def __init__(self, func):
        '''
        This class is to decorate on another function where you want to run in a new process
        with brand new environment variables (e.g. OMP/KMP, LD_PRELOAD, ...)
        an example to use this is
            ```python
            def throughput_helper(model, x):
                st = time.time()
                for _ in range(100):
                    model(x)
                return time.time() - st

            # make this wrapper
            # note: please name the new function a new func name.
            new_throughput_helper = spawn_new_process(throughput_helper)

            # this will run in current process
            duration = new_throughput_helper(model, x)

            # this will run in a new process with new env var effective
            duration = new_throughput_helper(model, x, env_var={"OMP_NUM_THREADS": "1",
                                                                "LD_PRELOAD": ...})
            ```
        '''
        self.func = func

    def __call__(self, *args, **kwargs):
        if "env_var" in kwargs:
            # check env_var should be a dict
            invalidInputError(isinstance(kwargs['env_var'], dict),
                              "env_var should be a dict")

            # prepare
            # 1. save the original env vars
            # 2. set new ones
            # 3. change the backend of multiprocessing from fork to spawn
            # 4. delete "env_var" from kwargs
            old_env_var = {}
            for key, value in kwargs['env_var'].items():
                old_env_var[key] = os.environ.get(key, "")
                os.environ[key] = value
            try:
                mp.set_start_method('spawn')
            except Exception:
                pass
            del kwargs["env_var"]

            # new process
            # block until return
            q = mp.Queue()
            new_func = _append_return_to_pipe(self.func)
            p = Process(target=new_func, args=(q,) + args, kwargs=kwargs)
            p.start()
            return_val = q.get()
            p.join()

            # recover
            for key, value in old_env_var.items():
                os.environ[key] = value

            return return_val
        else:
            return self.func(*args, **kwargs)


def get_default_args(func):
    """
    Check function `func` and get its arguments which has default value.

    :param func: Function to check.
    :return: A dict, contains arguments and their default values.
    """
    default_args = {}
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if param.default is not param.empty:
            default_args[param.name] = param.default
    return default_args


def compare_version(package: str, op: Callable, version: str,
                    use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))
