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
import re
from typing import Optional
from bigdl.dllib.utils.log4Error import *


def get_cgroup_cpuset():
    with open("/sys/fs/cgroup/cpuset/cpuset.cpus", "r") as f:
        content = f.readlines()
    cpu_set = []
    values = content[0].strip().split(",")
    for value in values:
        if "-" in value:
            # Parse the value like "2-4"
            start, end = value.split("-")
            cpu_set.extend([i for i in range(int(start), int(end) + 1)])
        else:
            cpu_set.append(int(value))
    return cpu_set


def get_cpu_info():
    cpuinfo = []
    args = ["lscpu", "--parse=CPU,Core,Socket"]
    lscpu_info = subprocess.check_output(args, universal_newlines=True).split("\n")

    # Get information about  cpu, core, socket and node
    for line in lscpu_info:
        pattern = r"^([\d]+,[\d]+,[\d]+)"
        regex_out = re.search(pattern, line)
        if regex_out:
            cpuinfo.append(regex_out.group(1).strip().split(","))

    get_physical_core = {}
    get_socket = {}

    for line in cpuinfo:
        int_line = [int(x) for x in line]
        l_id, p_id, s_id = int_line
        get_physical_core[l_id] = p_id
        get_socket[l_id] = s_id

    return get_physical_core, get_socket


def schedule_workers(num_workers: int, cores_per_worker: Optional[int] = None):

    # If we are in a docker container whose --cpuset-cpus are set,
    # we can get available cpus in /sys/fs/cgroup/cpuset/cpuset.cpus.
    # If we are not in a container, this just return all cpus.
    cpuset = get_cgroup_cpuset()
    cpuset = sorted(cpuset)
    l_core_to_p_core, l_core_to_socket = get_cpu_info()

    p2l = {}
    p_cores = set()
    for logical_core in cpuset:
        physical_core = l_core_to_p_core[logical_core]
        p_cores.add(physical_core)
        if physical_core not in p2l:
            p2l[physical_core] = logical_core
    p_cores = sorted(p_cores)

    if cores_per_worker is None:
        cores_per_worker = len(p_cores) // num_workers

    msg = "total number of cores requested must be smaller or" \
          " equal than the physical cores available"
    invalidInputError(cores_per_worker * num_workers <= len(p_cores), msg)

    schedule = []
    for i in range(num_workers):
        schedule.append([p2l[core] for core in p_cores[i*cores_per_worker:(i+1)*cores_per_worker]])

    return schedule
