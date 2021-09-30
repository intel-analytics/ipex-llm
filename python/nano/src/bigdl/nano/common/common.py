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
import sys
import subprocess
import logging
import warnings
from typing import Union, Dict, List, Optional
import numpy as np
import re

from bigdl.nano.common.cpu_schedule import schedule_workers


def check_avx512():
    cmd = "lscpu | grep avx512"
    try:
        subprocess.check_output(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        logging.warning("avx512 disabled, fall back to non-ipex mode.")
        return False


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


def init_nano(use_malloc: str = "tc", use_openmp: bool = True,
              print_environment: bool = False) -> None:
    """
    Configure necessary environment variables for jemalloc and openmp libraries.
    :param use_malloc: Allocator to be chosen, either "je" for jemalloc or "tc" for tcmalloc.
        default as tcmalloc.
    :param use_openmp: If this is set to True, then use intel openmp library. Otherwise disable
        openmp and related environment variables.
    :param print_environment: If this is set to True, print all environment variables after
        setting.
    :return: None
    """

    # Get a copy of os environment
    env_copy = os.environ.copy()

    if _env_variable_is_set("BIGDL_NANO_CHILD", env_copy):
        return

    # Find conda directory
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'"
                                            "| sed 's/.*:.//g'",
                                            shell=True).splitlines()[0].decode("utf-8")
    except subprocess.CalledProcessError:
        warnings.warn("Conda is not found on your computer.")

    conda_lib_dir = conda_dir + "/lib" if conda_dir is not None else None
    openmp_lib_dir = _find_library("libiomp5.so", conda_lib_dir)
    jemalloc_lib_dir = _find_library("libjemalloc.so", conda_lib_dir)
    tc_malloc_lib_dir = _find_library("libtcmalloc.so", conda_lib_dir)
    ld_preload_list = []

    # Detect Intel OpenMP library
    if openmp_lib_dir is not None:
        ld_preload_list.append(openmp_lib_dir)

        # Detect number of physical cores
        cpu_procs = schedule_workers(1)
        num_threads = len(cpu_procs[0])

        # Set environment variables
        if not _env_variable_is_set("OMP_NUM_THREADS", env_copy):
            env_copy["OMP_NUM_THREADS"] = str(num_threads)

        if not _env_variable_is_set("KMP_AFFINITY", env_copy):
            env_copy["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

        if not _env_variable_is_set("KMP_BLOCKTIME", env_copy):
            env_copy["KMP_BLOCKTIME"] = "1"
    else:
        warnings.warn("Intel OpenMP library (libiomp5.so) is not found.")

    # Detect jemalloc library
    if jemalloc_lib_dir is not None:
        ld_preload_list.append(jemalloc_lib_dir)

        if not _env_variable_is_set("MALLOC_CONF", env_copy):
            env_copy["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,"\
                "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
    else:
        warnings.warn("jemalloc library (libjemalloc.so) is nor found.")

    if tc_malloc_lib_dir is not None:
        ld_preload_list.append(tc_malloc_lib_dir)
    else:
        warnings.warn("tcmalloc library (libtcmalloc.so) is nor found.")

    # Set LD_PRELOAD
    if not _env_variable_is_set("LD_PRELOAD", env_copy):
        env_copy["LD_PRELOAD"] = " ".join(ld_preload_list)

    # Disable openmp or jemalloc according to options
    ld_preload = env_copy["LD_PRELOAD"].split(" ")
    if not use_openmp:
        env_copy.pop("OMP_NUM_THREADS")
        env_copy.pop("KMP_AFFINITY")
        env_copy.pop("KMP_BLOCKTIME")
        ld_preload = [lib for lib in ld_preload if "libiomp5.so" not in lib]

    if use_malloc is not "je":
        env_copy.pop("MALLOC_CONF")
        ld_preload = [lib for lib in ld_preload if "libjemalloc.so" not in lib]

    if use_malloc is not "tc":
        ld_preload = [lib for lib in ld_preload if "libtcmalloc.so" not in lib]

    env_copy["LD_PRELOAD"] = " ".join(ld_preload)
    env_copy["BIGDL_NANO_CHILD"] = "1"

    if print_environment:
        print(env_copy)

    if len(sys.argv) > 0 and len(sys.argv[0]) > 0:
        # Not in an interactive shell (sys.argv is not [""])
        os.execve(sys.executable, [sys.executable] + sys.argv, env_copy)
    else:
        os.execve(sys.executable, [sys.executable], env_copy)
