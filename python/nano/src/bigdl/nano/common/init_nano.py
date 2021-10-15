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


import argparse
import os
import subprocess
from typing import Union, Optional
import warnings

from bigdl.nano.common.cpu_schedule import schedule_workers, get_cpu_info

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--openmp", action="store_false", default=True,
                    help="Whether to disable Intel OpenMP.")
parser.add_argument("-j", "--jemalloc", action="store_false", default=True,
                    help="Whether to disable jemalloc.")
parser.add_argument("-v", "--verbose", action="store_true", default=False,
                    help="Whether to verbose about manipulations.")
parser.add_argument("-tf", "--tensorflow", action="store_true", default=False,
                    help="Whether to enable tensorflow optimaztion.")
args = parser.parse_args()
env_copy = os.environ.copy()


def _set_env_var(**kwargs):
    """Set the environment variables using anaconda."""
    for k, v in kwargs.items():
        v = str(v)

        cmd = ["conda", "env", "config", "vars", "set", f"{k}={v}"]
        subprocess.run(cmd, check=True, capture_output=True)
        if args.verbose:
            print(f"Set {k}={v}")


def _env_variable_is_set(variable: str) -> bool:
    """
    Return True if the environment variable is set by the user (i.e. set and not empty)
    :param variable: Name of the environment variable.
    """

    return variable in env_copy and len(env_copy[variable]) > 0


def _get_env_var(variable: str) -> Optional[str]:
    """Return the environment variable value, and return None if not found."""
    if _env_variable_is_set(variable):
        return env_copy[variable]
    else:
        return None


def _unset_env_var(*vs):
    """Unset the environment variables using conda.

    NOTE - conda can only unset env variables set by itself.
    """
    for v in vs:
        v = str(v)
        cmd = ["conda", "env", "config", "vars", "unset", v]
        subprocess.run(cmd, check=True, capture_output=True)
        if args.verbose:
            print(f"Unset {v}")


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
            warnings.warn("Some errors occurred while trying to find " + library_name)
        if len(res) > 0:
            return res[0].decode("utf-8")

    try:
        res = subprocess.check_output("find / -name " + library_name, shell=True,
                                      stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        warnings.warn("Some errors occurred while trying to find " + library_name)
    return res[0].decode("utf-8") if len(res) > 0 else None


def main():
    """Set up all the env variables."""
    use_jemalloc = args.jemalloc
    use_openmp = args.openmp
    enable_tf = args.tensorflow
    env_params = {}  # Environment variables to change

    # Find conda directory
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'"
                                            "| sed 's/.*:.//g'",
                                            shell=True).splitlines()[0].decode("utf-8")
        conda_env_name = conda_dir.split("/")[-1]
    except subprocess.CalledProcessError:
        raise LookupError("Conda is not found on your computer.")

    # Unset old conda env variables
    _unset_env_var(*["OMP_NUM_THREADS", "KMP_AFFINITY",
                   "KMP_BLOCKTIME", "MALLOC_CONF",
                     "LD_PRELOAD", "TF_ENABLE_ONEDNN_OPTS"])
    conda_lib_dir = conda_dir + "/lib" if conda_dir is not None else None
    openmp_lib_dir = _find_library("libiomp5.so", conda_lib_dir)
    jemalloc_lib_dir = _find_library("libjemalloc.so", conda_lib_dir)
    ld_preload_list = []

    # Detect Intel OpenMP library
    if use_openmp:
        if openmp_lib_dir is not None:
            ld_preload_list.append(openmp_lib_dir)

            # Detect number of physical cores
            cpu_procs = schedule_workers(1)
            _, get_socket = get_cpu_info()
            num_sockets = len(set(get_socket.values()))
            num_threads = len(cpu_procs[0]) // num_sockets if enable_tf else len(cpu_procs[0])

            # Set environment variables
            env_params["OMP_NUM_THREADS"] = str(num_threads)
            env_params["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
            env_params["KMP_BLOCKTIME"] = "1"

        else:
            warnings.warn("Intel OpenMP library (libiomp5.so) is not found.")

    # Detect jemalloc library
    if use_jemalloc:
        if jemalloc_lib_dir is not None:
            ld_preload_list.append(jemalloc_lib_dir)

            env_params["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,"\
                "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
        else:
            warnings.warn("jemalloc library (libjemalloc.so) is not found.")

    # Set LD_PRELOAD
    if ld_preload_list:
        env_params["LD_PRELOAD"] = " ".join(ld_preload_list)

    # Set tensorflow optimazion
    if enable_tf:
        env_params["TF_ENABLE_ONEDNN_OPTS"] = 1

    _set_env_var(**env_params)
    print("\nSucceed. To make your changes take effect please reactivate your environment:\n",
          "\n\tconda deactivate",
          "\n\tconda activate {}\n".format(conda_env_name),
          "\nTo inspect your environment variables set by this script:\n",
          "\n\tconda env config vars list -n {}\n".format(conda_env_name)
          )


if __name__ == "__main__":
    main()
