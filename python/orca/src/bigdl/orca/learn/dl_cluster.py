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

import ray
from bigdl.orca.cpu_info import schedule_workers
from bigdl.dllib.utils.log4Error import invalidInputError
import os
import sys
import logging
log = logging.getLogger(__name__)


class ClusterInfo:
    def ip_addr(self):
        return ray._private.services.get_node_ip_address()

    def set_cpu_affinity(self, core_list):
        proclist_str = f"[{','.join([str(i) for i in core_list])}]"
        os.environ["OMP_NUM_THREADS"] = str(len(core_list))
        os.environ["OMP_SCHEDULE"] = "STATIC"
        os.environ["OMP_PROC_BIND"] = "CLOSE"
        # KMP_AFFINITY works on intel openmp (intel tensorlow, intel pytorch/ipex)
        # GOMP_CPU_AFFINITY works on gomp (stock pytorch)
        # os.sched_setaffinity works on other threads (stock tensorflow)
        os.environ["KMP_AFFINITY"] = f"verbose,granularity=fine,proclist={proclist_str},explicit"
        os.environ["GOMP_CPU_AFFINITY"] = proclist_str
        os.sched_setaffinity(0, set(core_list))

    def disable_cpu_affinity(self, num_cores):
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["KMP_AFFINITY"] = "disabled"
        os.environ["OMP_PROC_BIND"] = "FALSE"

    def run(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def make_worker(worker_cls):
    class Worker(worker_cls, ClusterInfo):
        pass
    return Worker


class RayDLCluster:

    def __init__(self,
                 num_workers,
                 worker_cores,
                 worker_cls=None,
                 worker_param=None,
                 cpu_binding=True,
                 ):
        if not ray.is_initialized():
            invalidInputError(False,
                              "Ray is not initialize. Please initialize ray.")

        self.num_workers = num_workers
        self.worker_cores = worker_cores
        self.worker_cls = make_worker(worker_cls)
        self.work_param = worker_param
        if sys.platform == 'linux':
            self.cpu_binding = cpu_binding
        else:
            if cpu_binding:
                log.warn(f"cpu_binding is only support in linux, detectiong os {sys.platform}, "
                         "set cpu_binding to False")

            self.cpu_binding = False

        self.worker_class = ray.remote(num_cpus=self.worker_cores)(self.worker_cls)
        self.remote_workers = [self.worker_class.remote(**worker_param)
                               for i in range(0, self.num_workers)]

        if self.cpu_binding:
            hosts = ray.get([worker.ip_addr.remote() for worker in self.remote_workers])
            ip2workers = {}
            for ip, worker in zip(hosts, self.remote_workers):
                if ip not in ip2workers:
                    ip2workers[ip] = []
                ip2workers[ip].append(worker)
            ips = ip2workers.keys()

            cpu_binding_refs = []
            for ip in ips:
                ref = ip2workers[ip][0].run.remote(schedule_workers,
                                                   len(ip2workers[ip]),
                                                   self.worker_cores)
                cpu_binding_refs.append(ref)
            cpu_bindings = ray.get(cpu_binding_refs)

            result = []
            for ip, core_lists in zip(ips, cpu_bindings):
                for worker, core_list in zip(ip2workers[ip], core_lists):
                    log.debug(f"Setting thread affinity for worker in {ip}: {core_list}")
                    result.append(worker.set_cpu_affinity.remote(core_list))

            ray.get(result)
        else:
            ray.get([worker.disable_cpu_affinity.remote(self.worker_cores)
                     for worker in self.remote_workers])

    def get_workers(self):
        return self.remote_workers
