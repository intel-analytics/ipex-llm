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


from cpuinfo import CPUID


class ISAChecker:
    def __init__(self):
        cpuid = CPUID()
        self.flags = cpuid.get_flags(cpuid.get_max_extension_support())
        if self._avx_vnni(cpuid):
            self.flags.append('avxvnni')

    def _avx_vnni(self, cpuid):
        eax = cpuid._run_asm(
            b"\xB9\x01\x00\x00\x00",   # mov ecx, 0x1
            b"\xB8\x07\x00\x00\x00",   # mov eax, 0x7
            b"\x0f\xa2",               # cpuid
            b"\xC3"                    # ret
        )
        return ((0x10) & eax) != 0

    def check_avx(self):
        return 'avx' in self.flags

    def check_avx2(self):
        return 'avx2' in self.flags

    def check_avx_vnni(self):
        return 'avxvnni' in self.flags

    def check_avx512(self):
        return 'avx512f' in self.flags and \
               'avx512bw' in self.flags and \
               'avx512cd' in self.flags and \
               'avx512dq' in self.flags and \
               'avx512vl' in self.flags

    def check_avx512_vnni(self):
        return 'avx512vnni' in self.flags


isa_checker = ISAChecker()


def check_avx():
    return isa_checker.check_avx()


def check_avx2():
    return isa_checker.check_avx2()


def check_avx_vnni():
    return isa_checker.check_avx_vnni() and isa_checker.check_avx2()


def check_avx512():
    return isa_checker.check_avx512()


def check_avx512_vnni():
    return isa_checker.check_avx512_vnni() and isa_checker.check_avx512()


def is_server():
    return check_avx512_vnni()


# todo: use cpuid to check SPR
# note: now only SPR supports both avxvnni and avx512vnni
def is_spr():
    return check_avx_vnni() and check_avx512_vnni()
