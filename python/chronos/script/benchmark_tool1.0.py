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
import sys
import logging
import warnings
from typing import Union, Dict, List, Optional
import numpy as np
import re

from bigdl.nano.common.cpu_schedule import schedule_workers


def get_bytesize(bytes):
    """
    Scale bytes to its proper format ( B / KB / MB / GB / TB / PB )
    """
    factor = 1024
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if bytes < factor:
            return str(format(bytes,'.2f')) + unit
        bytes /= factor


def CPU_info():
    """
    Capture hardware information, such as CPU model, CPU informations, memory status
    """

    #information about CPU
    socket_num = int(subprocess.getoutput('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
    model_name = subprocess.getoutput('lscpu | grep "Model name"')
    model_name = model_name.partition(":")[2]

    print(">"*20,"Hardware Information",">"*20)
    print('CPU architecture:',platform.processor())
    print('CPU model name:',model_name.lstrip())
    print('Logical Core(s):',psutil.cpu_count()) 
    print('Physical Core(s):',psutil.cpu_count(logical=False)) 
    print('Physical Core(s) per socket:',int(psutil.cpu_count(logical=False)/socket_num))
    print('Socket(s):',socket_num)
    print('CPU usage:',str(psutil.cpu_percent()) + '%')
    print('CPU MHz:',format(psutil.cpu_freq().current,'.2f'))
    print('CPU max MHz:',format(psutil.cpu_freq().max,'.2f'))
    print('CPU min MHz:',format(psutil.cpu_freq().min,'.2f'))
    print('Total memory:',get_bytesize(psutil.virtual_memory().total))
    print('Available memory:',get_bytesize(psutil.virtual_memory().available))

    #support instruction set or not
    flag1 = int(subprocess.getoutput('cat /proc/cpuinfo | grep "flags"| sort -u | grep -c "avx512bw" '))
    if flag1 == 0:
        print('Support avx512bw:',"\033[0;31m\u2718\033[0m")
    else:
        print('Support avx512bw:',"\033[0;32m\u2714\033[0m")

    flag2 = int(subprocess.getoutput('cat /proc/cpuinfo | grep "flags"| sort -u | grep -c "avx512_bf16" '))
    if flag2 == 0:
        print('Support avx512_bf16:',"\033[0;31m\u2718\033[0m")
    else:
        print('Support avx512_bf16:',"\033[0;32m\u2714\033[0m")

    flag3 = int(subprocess.getoutput('cat /proc/cpuinfo | grep "flags"| sort -u | grep -c "avx512_vnni" '))
    if flag3 == 0:
        print('Support avx512_vnni:',"\033[0;31m\u2718\033[0m")
    else:
        print('Support avx512_vnni:',"\033[0;32m\u2714\033[0m")
    
    print("<"*20,"Hardware Information","<"*20,"\n")
    

def _env_variable_is_set(variable: str,
                         env: Union[Dict[str, str], os._Environ] = os.environ) -> bool:
    """
    Return True if the environment variable is set by the user (i.e. set and not empty)
    :param variable: Name of the environment variable.
    :param env: A dictionary showing the environment variable, default: os.environ
    """

    return variable in env and len(env[variable]) > 0


def _find_library(library_name: str, priority_dir: Union[str, None] = None) -> Union[str, None]:
    """
    Find the absolute path of the given library name. This function will search in the
    priority directory first, and if the library is not found, it will search the root
    directory. If the library is not found, the function will return None. If there
    are multiple paths available, return only one of the paths.
    :param library_name: The name of library to be found.
    :param priority_dir: A string indicating the absolute path of the directory that
        will be searched first. default: None.
    :return: the string of the absolute path of the library or None if the library is not found.
    """

    res = []
    if priority_dir is not None:
        try:
            res = subprocess.check_output("find " + priority_dir + " -name " + library_name,
                                          shell=True, stderr=subprocess.DEVNULL).splitlines()
        except Exception:
            warnings.warn(
                "Some errors occurred while trying to find " + library_name)
        if len(res) > 0:
            return res[0].decode("utf-8")

    try:
        res = subprocess.check_output("find / -name " + library_name, shell=True,
                                      stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        warnings.warn(
            "Some errors occurred while trying to find " + library_name)
    return res[0].decode("utf-8") if len(res) > 0 else None


def _find_path(path_name: str) -> bool:
    """
    Find whether .so files exist under the paths or not. This function will search the path one by one,
    and confirm whether libiomp5.so and libtcmalloc.so exist or not. If .so files can be found, return 
    True. Otherwise, return False.
    :param path_name: These paths to be found.
    :return: True(.so files can be found) or False(not all files can be found)
    """

    path_list = path_name.split(" ")
    libiomp5_flag = 0
    libtcmalloc_flag = 0

    for ipath in path_list:
        if os.path.exists(ipath):
            if ipath.endswith('libiomp5.so'):
                libiomp5_flag=1
            elif ipath.endswith('libtcmalloc.so'):
                libtcmalloc_flag=1

    return True if libiomp5_flag and libtcmalloc_flag else False
    


def check_nano(use_malloc: str = "tc", use_openmp: bool = True,
              print_environment: bool = False) -> None:
    """
    Check whether necessary environment variables are setted properly
    """
    # Get a copy of os environment
    env_copy = os.environ.copy()

    
    # Flags about the environment values are proper or not
    OpenMp_flag = 1
    tcmalloc_flag = 1
    TFsupport_flag = 1
    LD_PRELOAD_flag = 1

    #Output information
    openmp_name = ""
    tf_name = ""
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
    ld_preload_list = []

    # Detect Intel OpenMP library
    if use_openmp:
        if openmp_lib_dir is not None:
            ld_preload_list.append(openmp_lib_dir)

            # Detect number of physical cores
            cpu_procs = schedule_workers(1)
            num_threads = len(cpu_procs[0])

            # Check environment variables
            if not _env_variable_is_set("OMP_NUM_THREADS", env_copy) or env_copy["OMP_NUM_THREADS"] != str(num_threads):
                OpenMp_flag = 0
                openmp_name = openmp_name + " OMP_NUM_THREADS,"
                output_list.append("+"*50)
                output_list.append("The environment variable 'OMP_NUM_THREADS' is not proper.")
                output_list.append("Current value: " + env_copy["OMP_NUM_THREADS"])
                output_list.append("Suggested value: " + str(num_threads))
                output_list.append("The variable can be set by 'export OMP_NUM_THREADS=" + str(num_threads) + "'.")

            if not _env_variable_is_set("KMP_AFFINITY", env_copy) or env_copy["KMP_AFFINITY"] != "granularity=fine,compact,1,0":
                OpenMp_flag = 0
                openmp_name = openmp_name + " KMP_AFFINITY,"
                output_list.append("+"*50)
                output_list.append("The environment variable 'KMP_AFFINITY' is not proper.")
                output_list.append("Current value: " + env_copy["KMP_AFFINITY"])
                output_list.append("Suggested value: granularity=fine,compact,1,0")
                output_list.append("The variable can be set by 'export KMP_AFFINITY=granularity=fine,compact,1,0'.")

            if not _env_variable_is_set("KMP_BLOCKTIME", env_copy) or env_copy["KMP_BLOCKTIME"] != "1":
                OpenMp_flag = 0
                openmp_name = openmp_name + " KMP_BLOCKTIME,"
                output_list.append("+"*50)
                output_list.append("The environment variable 'KMP_BLOCKTIME' is not proper.")
                output_list.append("Current value: " + env_copy["KMP_BLOCKTIME"])
                output_list.append("Suggested value: 1")
                output_list.append("The variable can be set by 'export KMP_BLOCKTIME=1'.")
        else:
            output_list.append("Intel OpenMP library (libiomp5.so) is not found.")

    # Detect jemalloc library
    if use_malloc is "je":
        if jemalloc_lib_dir is not None:
            ld_preload_list.append(jemalloc_lib_dir)

            if not _env_variable_is_set("MALLOC_CONF", env_copy) or env_copy["MALLOC_CONF"] != "oversize_threshold:1,background_thread:true,"\
                    "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1":
                output_list.append("+"*50)
                output_list.append("The environment variable 'MALLOC_CONF' is not proper.")
                output_list.append("Current value: " + env_copy["MALLOC_CONF"])
                output_list.append("Suggested value: ","oversize_threshold:1,background_thread:true,"\
                    "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1")
                output_list.append("The variable can be set by 'export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1'.")
            
        else:
            output_list.append("jemalloc library (libjemalloc.so) is not found.")

    if use_malloc is "tc":
        if tc_malloc_lib_dir is not None:
            ld_preload_list.append(tc_malloc_lib_dir)
        else:
            tcmalloc_flag = 0
            output_list.append("tcmalloc library (libtcmalloc.so) is not found.")
    

    # Check TF_support
    if not _env_variable_is_set("TF_ENABLE_ONEDNN_OPTS", env_copy) or env_copy["TF_ENABLE_ONEDNN_OPTS"] != "1":
        TFsupport_flag = 0
        tf_name = tf_name + " TF_ENABLE_ONEDNN_OPTS,"
        output_list.append("+"*50)
        output_list.append("The environment variable 'TF_ENABLE_ONEDNN_OPTS' is not proper.")
        output_list.append("Current value: " + env_copy["TF_ENABLE_ONEDNN_OPTS"])
        output_list.append("Suggested value: 1")
        output_list.append("The variable can be set by 'export TF_ENABLE_ONEDNN_OPTS=1'.")

    if not _env_variable_is_set("ENABLE_TF_OPTS", env_copy) or env_copy["ENABLE_TF_OPTS"] != "1":
        TFsupport_flag = 0
        tf_name = tf_name + " ENABLE_TF_OPTS,"
        output_list.append("+"*50)
        output_list.append("The environment variable 'ENABLE_TF_OPTS' is not proper.")
        output_list.append("Current value: " + str(env_copy["ENABLE_TF_OPTS"]))
        output_list.append("Suggested value: 1")
        output_list.append("The variable can be set by 'export ENABLE_TF_OPTS=1'.")

    if not _env_variable_is_set("NANO_TF_INTER_OP", env_copy) or env_copy["NANO_TF_INTER_OP"] != "1":
        TFsupport_flag = 0
        tf_name = tf_name + " NANO_TF_INTER_OP,"
        output_list.append("+"*50)
        output_list.append("The environment variable 'NANO_TF_INTER_OP' is not proper.")
        output_list.append("Current value: " + env_copy["NANO_TF_INTER_OP"])
        output_list.append("Suggested value: 1")
        output_list.append("The variable can be set by 'export NANO_TF_INTER_OP=1'.")


    # Check LD_PRELOAD
    if not use_openmp:
        ld_preload_list = [lib for lib in ld_preload_list if "libiomp5.so" not in lib]

    if use_malloc is not "je":
        ld_preload_list = [lib for lib in ld_preload_list if "libjemalloc.so" not in lib]

    if use_malloc is not "tc":
        ld_preload_list = [lib for lib in ld_preload_list if "libtcmalloc.so" not in lib]

    if not _env_variable_is_set("LD_PRELOAD", env_copy) or not _find_path(env_copy["LD_PRELOAD"]):
        LD_PRELOAD_flag = 0
        output_list.append("+"*50)
        output_list.append("The environment variable 'LD_PRELOAD' is not proper.")
        output_list.append("Current value: " + env_copy["LD_PRELOAD"])
        output_list.append("Suggested value: " + " ".join(ld_preload_list))
        output_list.append("The variable can be set by 'export LD_PRELOAD=" + " ".join(ld_preload_list) + "'.")

    
    # Output overview
    print(">"*20,"Environment Variables",">"*20)
    if LD_PRELOAD_flag == 0:
        print('LD_PRELOAD',"\033[0;31m not enabled\033[0m","\033[0;31m\u2718\033[0m")
    else:
        print('LD_PRELOAD enabled ',"\033[0;32m\u2714\033[0m")
    
    if tcmalloc_flag == 0:
        print('tcmalloc',"\033[0;31m not enabled\033[0m","\033[0;31m\u2718\033[0m")
    else:
        print('tcmalloc enabled ',"\033[0;32m\u2714\033[0m")
    
    if OpenMp_flag == 0:
        print('Intel OpenMp:',openmp_name[:-1],"\033[0;31m not enabled\033[0m","\033[0;31m\u2718\033[0m") 
    else:
        print('Intel OpenMp enabled ',"\033[0;32m\u2714\033[0m")

    if TFsupport_flag == 0:
        print('TF:',tf_name[:-1],"\033[0;31m not enabled\033[0m","\033[0;31m\u2718\033[0m")
    else:
        print('TF enabled ',"\033[0;32m\u2714\033[0m")


    # Output suggestions
    if output_list is not None:
        for info in output_list:
            print(info)
    
    print("<"*20,"Environment Variables","<"*20,"\n")



if __name__ == '__main__':
    CPU_info()
    check_nano("tc",True,False)