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


import cpuinfo


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
