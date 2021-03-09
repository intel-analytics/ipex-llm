#
# Copyright 2018 Analytics Zoo Authors.
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
import os


class HorovodWorker:

    def ip_addr(self):
        import ray
        return ray._private.services.get_node_ip_address()

    def set_gloo_iface(self):
        ip_addr = self.ip_addr()
        import psutil
        import socket
        iface_name = None
        for intf, intf_addresses in psutil.net_if_addrs().items():
            for addr in intf_addresses:
                if addr.family == socket.AF_INET and addr.address == ip_addr:
                    iface_name = intf
        assert iface_name is not None, "Cannot find network interface with ip {}".format(ip_addr)

        os.environ["HOROVOD_GLOO_IFACE"] = iface_name
        return iface_name

    def run(self, env, func):
        import os
        os.environ.update(env)
        return func()


def _hosts_to_hosts_spec(hosts):
    host_to_size = {}
    host_and_rank_to_worker_idx = {}
    for i, host in enumerate(hosts):
        if host not in host_to_size:
            host_to_size[host] = 0
        else:
            host_to_size[host] = host_to_size[host] + 1
        host_and_rank_to_worker_idx[(host, host_to_size[host])] = i

    for key in host_to_size:
        host_to_size[key] += 1

    hosts_spec = ["{}:{}".format(key, host_to_size[key]) for key in host_to_size]
    return hosts_spec, host_and_rank_to_worker_idx, host_to_size


def make_worker(worker_cls, HorovodWorker):
    if worker_cls is None:
        return HorovodWorker
    if issubclass(worker_cls, HorovodWorker):
        return worker_cls

    class Worker(worker_cls, HorovodWorker):
        pass
    return Worker


def get_horovod_version():
    import horovod
    major, minor, patch = horovod.__version__.split(".")
    return int(major), int(minor), int(patch), horovod.__version__


class HorovodRayRunner:

    # todo check whether horovod is built with gloo
    def __init__(self, ray_ctx, worker_cls=None, worker_param=None, workers_per_node=1):
        self.cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
        self.num_nodes = ray_ctx.num_ray_nodes * workers_per_node
        if worker_param is None:
            worker_param = {}
        worker_cls = make_worker(worker_cls, HorovodWorker)
        self.worker_class = ray.remote(num_cpus=self.cores_per_node)(worker_cls)
        self.remote_workers = [self.worker_class.remote(**worker_param)
                               for i in range(0, self.num_nodes)]
        hosts = ray.get([worker.ip_addr.remote() for worker in self.remote_workers])
        hosts_spec, name_rank_to_id, host_to_size = _hosts_to_hosts_spec(hosts)

        major, minor, patch, version_str = get_horovod_version()

        if major == 0 and minor < 19:
            raise RuntimeError(f"We only support horovod versions newer "
                               f"than 0.19.0, but got {version_str}")
        if major == 0 and minor == 19:
            from horovod.run.gloo_run import RendezvousServer, _allocate
            self.host_alloc_plan = _allocate(",".join(hosts_spec), self.num_nodes)
            self.global_rendezv = RendezvousServer(True)
            global_rendezv_port = self.global_rendezv.start_server(self.host_alloc_plan)
        else:
            from horovod.runner.gloo_run import RendezvousServer, parse_hosts, get_host_assignments
            self.host_alloc_plan = get_host_assignments(parse_hosts(",".join(hosts_spec)),
                                                        self.num_nodes)
            self.global_rendezv = RendezvousServer(True)
            global_rendezv_port = self.global_rendezv.start()
            self.global_rendezv.init(self.host_alloc_plan)

        driver_ip = ray._private.services.get_node_ip_address()

        common_envs = {
            "HOROVOD_GLOO_RENDEZVOUS_ADDR": driver_ip,
            "HOROVOD_GLOO_RENDEZVOUS_PORT": str(global_rendezv_port),
            "HOROVOD_CONTROLLER": "gloo",
            "HOROVOD_CPU_OPERATIONS": "gloo",
            "PYTHONUNBUFFERED": '1',
            "OMP_NUM_THREADS": str(self.cores_per_node)
        }

        for key in os.environ:
            if key.startswith("HOROVOD"):
                common_envs[key] = os.environ[key]

        # todo support other Horovod envs
        self.per_worker_envs = [common_envs.copy() for _ in range(self.num_nodes)]
        for alloc_info in self.host_alloc_plan:
            key = (alloc_info.hostname, alloc_info.local_rank)
            local_envs = self.per_worker_envs[name_rank_to_id[key]]
            local_envs["HOROVOD_HOSTNAME"] = str(alloc_info.hostname)
            local_envs["HOROVOD_RANK"] = str(alloc_info.rank)
            local_envs["HOROVOD_SIZE"] = str(alloc_info.size)
            local_envs["HOROVOD_LOCAL_RANK"] = str(alloc_info.local_rank)
            local_envs["HOROVOD_LOCAL_SIZE"] = str(alloc_info.local_size)
            local_envs["HOROVOD_CROSS_RANK"] = str(alloc_info.cross_rank)
            local_envs["HOROVOD_CROSS_SIZE"] = str(alloc_info.cross_size)

        ray.get([worker.set_gloo_iface.remote() for worker in self.remote_workers])
        self.run(lambda: print("horovod worker initialized"))

    def run(self, func):
        return ray.get([self.remote_workers[i].run.remote(self.per_worker_envs[i], func)
                       for i in range(self.num_nodes)])
