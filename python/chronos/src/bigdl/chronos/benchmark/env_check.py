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


import psutil
import platform
import subprocess
import os
import logging
import warnings

from .utils_env import get_bytesize, _find_path, get_nano_env_var
from bigdl.nano.utils.common import _env_variable_is_set, _find_library
from psutil import cpu_count


def get_CPU_info():
    """
    Capture hardware information, such as CPU model, CPU informations, memory status
    """

    # information about CPU
    socket_num = int(subprocess.getoutput('cat /proc/cpuinfo | grep "physical id" | '
                                          'sort -u | wc -l'))
    model_name = subprocess.getoutput('lscpu | grep "Model name"')
    model_name = model_name.partition(":")[2]

    print(">"*20, "Hardware Information", ">"*20)
    print('\033[1m\tCPU architecture\033[0m:', platform.processor())
    print('\033[1m\tCPU model name\033[0m:', model_name.lstrip())
    print('\033[1m\tLogical Core(s)\033[0m:', cpu_count())
    print('\033[1m\tPhysical Core(s)\033[0m:', cpu_count(logical=False))
    print('\033[1m\tPhysical Core(s) per socket\033[0m:', int(cpu_count(logical=False)/socket_num))
    print('\033[1m\tSocket(s)\033[0m:', socket_num)
    print('\033[1m\tCPU usage\033[0m:', str(psutil.cpu_percent()) + '%')
    print('\033[1m\tCPU MHz\033[0m:', format(psutil.cpu_freq().current, '.2f'))
    print('\033[1m\tCPU max MHz\033[0m:', format(psutil.cpu_freq().max, '.2f'))
    print('\033[1m\tCPU min MHz\033[0m:', format(psutil.cpu_freq().min, '.2f'))
    print('\033[1m\tTotal memory\033[0m:', get_bytesize(psutil.virtual_memory().total))
    print('\033[1m\tAvailable memory\033[0m:', get_bytesize(psutil.virtual_memory().available))

    # support instruction set or not
    disabled_logo = "\033[0;31m\u2718\033[0m"
    abled_logo = "\033[0;32m\u2714\033[0m"

    for flag in ["avx512f", "avx512_bf16", "avx512_vnni"]:
        flag_enabled = int(subprocess.getoutput(f'lscpu | grep -c {flag} '))
        if flag_enabled:
            print("\033[1m\tSupport\033[0m", flag, ":", abled_logo)
        else:
            print("\033[1m\tSupport\033[0m", flag, ":", disabled_logo)

    print("<"*20, "Hardware Information", "<"*20, "\n")


def check_nano_env(use_malloc: str = "tc", use_openmp: bool = True) -> None:
    """
    Check whether necessary environment variables are setted properly
    """
    # Get a copy of os environment
    env_copy = os.environ.copy()
    # Get the proper environment
    correct_env = get_nano_env_var()

    # Flags about the environment values are proper or not
    flag = {"LD_PRELOAD": 1, "tcmalloc": 1, "Intel OpenMp": 1, "TF": 1}

    # Output information
    name = {"LD_PRELOAD": "", "tcmalloc": "", "Intel OpenMp": ": ", "TF": ": "}
    output_list = []

    # Find conda directory
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'"
                                            "| sed 's/.*:.//g'",
                                            shell=True).splitlines()[0].decode("utf-8")
        conda_env_name = conda_dir.split("/")[-1]
    except subprocess.CalledProcessError:
        warnings.warn("Conda is not found on your computer.")

    conda_lib_dir = conda_dir + "/lib" if conda_dir is not None else None
    openmp_lib_dir = _find_library("libiomp5.so", conda_lib_dir)
    jemalloc_lib_dir = _find_library("libjemalloc.so", conda_lib_dir)
    tc_malloc_lib_dir = _find_library("libtcmalloc.so", conda_lib_dir)

    # Detect Intel OpenMP library
    if use_openmp:
        if openmp_lib_dir is not None:
            # Check environment variables
            for var in ["OMP_NUM_THREADS", "KMP_AFFINITY", "KMP_BLOCKTIME"]:
                if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
                    flag["Intel OpenMp"] = 0
                    name["Intel OpenMp"] = name["Intel OpenMp"] + var + " "
                    output_list.append("export " + var + "=" + correct_env[var])
        else:
            output_list.append("Intel OpenMP library (libiomp5.so) is not found.")

    # Detect jemalloc library
    if use_malloc is "je":
        if jemalloc_lib_dir is not None:
            if (not _env_variable_is_set("MALLOC_CONF", env_copy) or
               env_copy["MALLOC_CONF"] != correct_env["MALLOC_CONF"]):
                output_list.append("export MALLOC_CONF=oversize_threshold:1,background_thread:"
                                   "true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1")
        else:
            output_list.append("jemalloc library (libjemalloc.so) is not found.")

    if use_malloc is "tc":
        if tc_malloc_lib_dir is None:
            flag["tcmalloc"] = 0
            output_list.append("tcmalloc library (libtcmalloc.so) is not found.")

    # Check TF_support
    for var in ["TF_ENABLE_ONEDNN_OPTS"]:
        if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
            flag["TF"] = 0
            name["TF"] = name["TF"] + var + " "
            output_list.append("export " + var + "=" + correct_env[var])

    # Check LD_PRELOAD
    if not _env_variable_is_set("LD_PRELOAD", env_copy) or not _find_path(env_copy["LD_PRELOAD"]):
        flag["LD_PRELOAD"] = 0
        output_list.append("export LD_PRELOAD=" + correct_env["LD_PRELOAD"])

    # Output overview
    print(">"*20, "Environment Variables", ">"*20)
    disabled_logo = "\033[0;31mnot enabled \033[0m" + "\033[0;31m\u2718\033[0m"
    abled_logo = "\033[0;32m enabled \033[0m" + "\033[0;32m\u2714\033[0m"
    for category in ["LD_PRELOAD", "tcmalloc", "Intel OpenMp", "TF"]:
        if flag[category] == 0:
            print(f"\033[1m\t{category}\033[0m", name[category], disabled_logo)
        else:
            print(f"\033[1m\t{category}\033[0m", abled_logo)

    # Output suggestions
    if output_list != []:
        print(" ")
        print("+" * 20, "Suggested change: ", "+" * 20)
        for info in output_list:
            print(info)
        print("+" * 60, "\n")

    print("<"*20, "Environment Variables", "<"*20, "\n")
