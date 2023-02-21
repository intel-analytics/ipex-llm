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
import platform
import cpuinfo
from bigdl.nano.utils.common import invalidInputError


def get_cgroup_cpuset():
    cpu_set = list(range(cpuinfo.get_cpu_info()["count"]))
    return cpu_set


def get_cpu_info():
    get_physical_core = {}
    get_socket = {}
    if platform.system() == "Windows":
        get_physical_core_args = ["wmic", "CPU",
                                  "GET", "NumberOfCores", "/VALUE"]
        physical_cores = int(subprocess.check_output(
            get_physical_core_args, universal_newlines=True).splitlines()[4].split("=")[1])
        get_logical_core_args = ["wmic", "CPU", "GET",
                                 "NumberOfLogicalProcessors", "/VALUE"]
        logical_cores = int(subprocess.check_output(
            get_logical_core_args, universal_newlines=True).splitlines()[4].split("=")[1])
        for i in range(logical_cores):
            # We don't know how to query which cores are physical cores on Windows,
            # so we use all logical cores
            get_physical_core[i] = i
            get_socket[i] = 0
    elif platform.system() == "Darwin":
        # We cannot query which cores are physical cores on MacOS,
        # so we use all logical cores
        get_logical_core_num_args = ["sysctl", "-n", "hw.ncpu"]
        logical_core_num = int(subprocess.check_output(get_logical_core_num_args,
                                                       universal_newlines=True))
        for i in range(logical_core_num):
            get_physical_core[i] = i
            get_socket[i] = 0
    else:
        cpuinfo = []

        args = ["lscpu", "--parse=CPU,Core,Socket"]
        lscpu_info = subprocess.check_output(
            args, universal_newlines=True).split("\n")

        # Get information about  cpu, core, socket and node
        for line in lscpu_info:
            pattern = r"^([\d]+,[\d]+,[\d]+)"
            regex_out = re.search(pattern, line)
            if regex_out:
                cpuinfo.append(regex_out.group(1).strip().split(","))

        for line in cpuinfo:
            int_line = [int(x) for x in line]
            l_id, p_id, s_id = int_line
            get_physical_core[l_id] = p_id
            get_socket[l_id] = s_id

    return get_physical_core, get_socket


def schedule_workers(num_workers: int,
                     cores_per_worker: Optional[int] = None):

    l_core_to_p_core, l_core_to_socket = get_cpu_info()
    try:
        cpuset = get_cgroup_cpuset()
    except Exception as _e:     # failed to query cgroup info
        cpuset = list(l_core_to_p_core.keys())
    cpuset = sorted(cpuset)

    p2l = {}
    p_cores_set = set()
    for logical_core in cpuset:
        physical_core = l_core_to_p_core[logical_core]
        p_cores_set.add(physical_core)
        if physical_core not in p2l:
            p2l[physical_core] = logical_core
    p_cores = sorted(p_cores_set)

    if cores_per_worker is None:
        cores_per_worker = len(p_cores) // num_workers
        invalidInputError(cores_per_worker > 0,
                          "The number of processes must be less "
                          "or equal to the number of physical cores")
    else:
        invalidInputError(cores_per_worker > 0,
                          "`cores_per_worker` must be greater than 0")

    msg = "total number of cores requested must be smaller or" \
          " equal than the physical cores available"
    invalidInputError(cores_per_worker * num_workers <= len(p_cores), msg)

    schedule = []
    for i in range(num_workers):
        schedule.append([p2l[core] for core in
                         p_cores[i * cores_per_worker:(i + 1) * cores_per_worker]])

    return schedule


def schedule_processors(num_workers: int,
                        cores_per_worker: Optional[int] = None):

    import os
    cpu_procs = schedule_workers(num_workers, cores_per_worker)

    envs = []
    if cpu_procs[0]:
        for i in range(num_workers):
            env = os.environ.copy()

            env.update({
                "KMP_AFFINITY": f"granularity=fine,proclist"
                                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                "PROCESS_IDX": str(i),
            })
            envs.append(env)
    else:
        for i in range(num_workers):
            env = os.environ.copy()

            env.update({
                "KMP_AFFINITY": "disabled",
                "OMP_NUM_THREADS": "1",
                "PROCESS_IDX": str(i)
            })
            envs.append(env)
    return envs
