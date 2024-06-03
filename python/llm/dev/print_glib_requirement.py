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

import subprocess
from packaging.version import Version
import pathlib


SUPPORTED_FAMILIES = ("llama", "bloom", "gptneox", "starcoder")


def _check_version(filename, flag="GLIBC"):
    subfile = None
    if flag == "GLIBC":
        subfile = _check_glibc_version(filename)
    if flag == "GLIBCXX":
        subfile = _check_glibcxx_version(filename)
    max_version = None
    if subfile:
        for version_string in subfile.split():
            try:
                version = Version(version_string.split("_")[1])
                if max_version is None or version > max_version:
                    max_version = version
            except Exception:
                pass
    return max_version


def _check_glibc_version(filename):
    cmd = f"strings {filename} | grep -v '@' | grep -v \"GLIBCXX\" | grep GLIBC"
    subfile = subprocess.getoutput(cmd)
    return subfile


def _check_glibcxx_version(filename):
    cmd = f"strings {filename} | grep -v '@' | grep \"GLIBCXX\""
    subfile = subprocess.getoutput(cmd)
    return subfile


if __name__ == "__main__":
    from bigdl import llm
    _base_dir = pathlib.Path(llm.__file__).parent.resolve() / "libs"

    requirements = {}

    for family in SUPPORTED_FAMILIES:
        requirements[family] = {}
        avx512_lib_c = _check_version(f"{_base_dir}/lib{family}_avx512.so", flag="GLIBC")
        avx512_main_c = _check_version(f"{_base_dir}/main-{family}_avx512", flag="GLIBC")
        requirements[family]["avx512_glibc"] = max(avx512_lib_c, avx512_main_c)

        avx512_lib_cxx = _check_version(f"{_base_dir}/lib{family}_avx512.so", flag="GLIBCXX")
        avx512_main_cxx = _check_version(f"{_base_dir}/main-{family}_avx512", flag="GLIBCXX")
        requirements[family]["avx512_glibcxx"] = max(avx512_lib_cxx, avx512_main_cxx)

        avx2_lib_c = _check_version(f"{_base_dir}/main-{family}_avx2", flag="GLIBC")
        avx2_main_c = _check_version(f"{_base_dir}/lib{family}_avx2.so", flag="GLIBC")
        requirements[family]["avx2_glibc"] = max(avx2_lib_c, avx2_main_c)

        avx2_lib_cxx = _check_version(f"{_base_dir}/lib{family}_avx2.so", flag="GLIBCXX")
        avx2_main_cxx = _check_version(f"{_base_dir}/main-{family}_avx2", flag="GLIBCXX")
        requirements[family]["avx2_glibcxx"] = max(avx2_lib_cxx, avx2_main_cxx)

        quantize_c = _check_version(f"{_base_dir}/quantize-{family}", flag="GLIBC")
        quantize_cxx = _check_version(f"{_base_dir}/quantize-{family}", flag="GLIBCXX")
        requirements[family]["quantize_glibc"] = quantize_c
        requirements[family]["quantize_glibcxx"] = quantize_cxx

    for family in SUPPORTED_FAMILIES:
        print(family)
        print(requirements[family])
