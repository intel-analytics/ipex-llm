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
from horovod.run.gloo_run import RendezvousServer, _allocate
from horovod.run.driver import driver_service
from horovod.run.task_fn import _task_fn
from horovod.run.common.util import settings as hvd_settings
from horovod.run.common.util import timeout, secret
from horovod.run.task import task_service


def make_horovod_worker(cores_per_node):

    # todo how to make user func honor this resource restriction
    @ray.remote(num_cpus=cores_per_node)
    class HorovodWorker:

        def hostname(self):
            import socket
            return socket.gethostname()

        def run(self, env, func):
            import os
            os.environ.update(env)
            return func()

        def task_fn(self, index, driver_addresses, settings):
            _task_fn(index, driver_addresses, settings)

    return HorovodWorker


def _get_driver_ip(common_intfs):
    from socket import AF_INET
    from psutil import net_if_addrs
    iface = list(common_intfs)[0]
    driver_ip = None
    for addr in net_if_addrs()[iface]:
        if addr.family == AF_INET:
            driver_ip = addr.address

    if not driver_ip:
        raise RuntimeError(
            'Cannot find an IPv4 address of the common interface.')

    return driver_ip


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


def _launch_task_servers(all_host_names, host_rank_to_id, driver_addresses, settings, workers):

    result_ids = []
    for index in range(len(all_host_names)):
        host_name = all_host_names[index]
        worker = workers[host_rank_to_id[(host_name, 0)]]

        result_id = worker.task_fn.remote(index, driver_addresses, settings)
        result_ids.append(result_id)

    return result_ids


def _find_common_network_interface(host_to_size, host_rank_to_id, workers, settings):
    all_host_names = [k for k in host_to_size]
    driver = driver_service.HorovodRunDriverService(len(all_host_names), settings.key, settings.nic)

    _launch_task_servers(all_host_names, host_rank_to_id, driver.addresses(), settings, workers)

    # the following code is copied and modified from horovod.run._driver_fn
    try:
        # wait for all the hosts to register with the service service.
        if settings.verbose >= 2:
            print('Waiting for the hosts to acknowledge.')
        driver.wait_for_initial_registration(settings.timeout)
        tasks = [
            task_service.HorovodRunTaskClient(
                index,
                driver.task_addresses_for_driver(index),
                settings.key,
                settings.verbose) for index in range(
                settings.num_hosts)]
        # Notify all the drivers that the initial registration is complete.
        for task in tasks:
            task.notify_initial_registration_complete()
        if settings.verbose >= 2:
            print('Notified all the hosts that the registration is complete.')
        # Each worker should probe the interfaces of the next worker in a ring
        # manner and filter only the routed ones -- it should filter out
        # interfaces that are not really connected to any external networks
        # such as lo0 with address 127.0.0.1.
        if settings.verbose >= 2:
            print('Waiting for hosts to perform host-to-host '
                  'interface checking.')
        driver.wait_for_task_to_task_address_updates(settings.timeout)
        if settings.verbose >= 2:
            print('Host-to-host interface checking successful.')
        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, settings.num_hosts):
            common_intfs.intersection_update(
                driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception(
                'Unable to find a set of common task-to-task communication '
                'interfaces: %s'
                % [(index, driver.task_addresses_for_tasks(index))
                   for index in range(settings.num_hosts)])
        return common_intfs
    finally:
        driver.shutdown()


class HorovodRayTrainer:

    # todo check whether horovod is built with gloo
    def __init__(self, ray_ctx, verbose=None, start_timeout=None):

        self.cores_per_node = ray_ctx.ray_node_cpu_cores
        self.num_nodes = ray_ctx.num_ray_nodes
        self.worker_class = make_horovod_worker(self.cores_per_node)
        self.remote_workers = [self.worker_class.remote() for i in range(0, self.num_nodes)]

        hosts = ray.get([worker.hostname.remote() for worker in self.remote_workers])
        hosts_spec, name_rank_to_id, host_to_size = _hosts_to_hosts_spec(hosts)
        self.host_alloc_plan = _allocate(",".join(hosts_spec), self.num_nodes)
        global_rendezv = RendezvousServer(True)
        global_rendezv_port = global_rendezv.start_server(self.host_alloc_plan)

        if start_timeout is None:
            start_timeout = int(os.getenv('HOROVOD_START_TIMEOUT', '30'))

        tmout = timeout.Timeout(start_timeout,
                                message='Timed out waiting for {activity}. Please '
                                        'check connectivity between servers. You '
                                        'may need to increase the --start-timeout '
                                        'parameter if you have too many servers.')

        all_host_names = [k for k in host_to_size]

        settings = hvd_settings.Settings(verbose=2 if verbose else 0,
                                         key=secret.make_secret_key(),
                                         timeout=tmout,
                                         num_hosts=len(all_host_names),
                                         num_proc=self.num_nodes,
                                         hosts=",".join(hosts_spec))

        common_intfs = _find_common_network_interface(host_to_size, name_rank_to_id,
                                                      self.remote_workers, settings)
        iface = list(common_intfs)[0]
        driver_ip = _get_driver_ip([iface])

        common_envs = {
            "HOROVOD_GLOO_RENDEZVOUS_ADDR": driver_ip,
            "HOROVOD_GLOO_RENDEZVOUS_PORT": str(global_rendezv_port),
            "HOROVOD_CONTROLLER": "gloo",
            "HOROVOD_CPU_OPERATIONS": "gloo",
            "HOROVOD_GLOO_IFACE": iface,
            "PYTHONUNBUFFERED": '1',
        }

        for key in os.environ:
            if key.startswith("HOROVOD"):
                common_envs[key] = os.environ[key]

        # todo support other Horovod envs
        self.per_worker_envs = [common_envs.copy() for _ in range(self.num_nodes)]
        for alloc_info in self.host_alloc_plan:
            key = (alloc_info.hostname, alloc_info.local_rank)
            local_envs = self.per_worker_envs[name_rank_to_id[key]]
            local_envs["HOROVOD_RANK"] = str(alloc_info.rank)
            local_envs["HOROVOD_SIZE"] = str(alloc_info.size)
            local_envs["HOROVOD_LOCAL_RANK"] = str(alloc_info.local_rank)
            local_envs["HOROVOD_LOCAL_SIZE"] = str(alloc_info.local_size)
            local_envs["HOROVOD_CROSS_RANK"] = str(alloc_info.cross_rank)
            local_envs["HOROVOD_CROSS_SIZE"] = str(alloc_info.cross_size)

    def train(self, func):
        ray.wait([self.remote_workers[i].run.remote(self.per_worker_envs[i], func)
                  for i in range(self.num_nodes)])
