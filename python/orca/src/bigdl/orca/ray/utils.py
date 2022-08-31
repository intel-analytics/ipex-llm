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

import re
from bigdl.dllib.utils.log4Error import invalidInputError


def to_list(input):
    if isinstance(input, (list, tuple)):
        return list(input)
    else:
        return [input]


def resource_to_bytes(resource_str):
    if not resource_str:
        return resource_str
    matched = re.compile("([0-9]+)([a-z]+)?").match(resource_str.lower())
    fraction_matched = re.compile("([0-9]+\\.[0-9]+)([a-z]+)?").match(resource_str.lower())
    if fraction_matched:
        invalidInputError(False,
                          "Fractional values are not supported. Input"
                          " was: {}".format(resource_str))
    try:
        value = int(matched.group(1))
        postfix = matched.group(2)
        if postfix == 'b':
            value = value
        elif postfix == 'k':
            value = value * 1000
        elif postfix == "m":
            value = value * 1000 * 1000
        elif postfix == 'g':
            value = value * 1000 * 1000 * 1000
        else:
            invalidInputError(False,
                              "Not supported type: {}".format(resource_str))
        return value
    except Exception:
        invalidInputError(False,
                          "Size must be specified as bytes(b),"
                          "kilobytes(k), megabytes(m), gigabytes(g). "
                          "E.g. 50b, 100k, 250m, 30g")


def is_local(sc):
    master = sc.getConf().get("spark.master")
    return master == "local" or master.startswith("local[")


def get_parent_pid(pid):
    import psutil
    cur_proc = psutil.Process(pid)
    return cur_proc.ppid()
