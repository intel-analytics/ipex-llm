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
import subprocess
import warnings
from typing import Union, Dict


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
