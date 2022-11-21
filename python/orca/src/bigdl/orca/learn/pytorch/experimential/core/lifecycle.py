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

from abc import abstractmethod, ABCMeta
import ray
from bigdl.orca.learn.pytorch.utils import find_free_port
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


class LifeCycle(metaclass=ABCMeta):
    def get_node_ip_port(self):
        ip = self.get_node_ip()
        port = find_free_port()
        return ip, port

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray._private.services.get_node_ip_address()

    def setup(self, cores_per_node):
        import torch
        torch.set_num_threads(cores_per_node)

    def setup_torch_distribute(self, tcp_store_host, tcp_store_port, world_rank,
                               world_size):
        self._init_torch_ddp(tcp_store_host, tcp_store_port, world_rank,
                             world_size)

        self.setup_components()

    @abstractmethod
    def setup_components(self):
        """Runs the creator functions without any distributed coordination."""

        # training_models = [
        #     DistributedDataParallel(model)
        #     for model in self.models
        # ]
        # self.setup_operator(training_models)
        pass

    def _init_torch_ddp(self, tcp_store_host, tcp_store_port, world_rank,
                        world_size):
        import torch.distributed as dist
        client_store = dist.TCPStore(tcp_store_host, tcp_store_port, -1, False)
        dist.init_process_group(
            backend="gloo",
            store=client_store,
            rank=world_rank,
            world_size=world_size)
        self.backend = "torch-distributed"
        self.rank = world_rank
        self.size = world_size

    def with_sampler(self, loader):
        self.logger.debug("Wrapping DistributedSampler on DataLoader")
        data_loader_args = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": False,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "sampler": DistributedSampler(loader.dataset,
                                          num_replicas=self.size,
                                          rank=self.rank)
        }
        return DataLoader(**data_loader_args)

    @staticmethod
    def should_wrap_dataloader(loader):
        from torch.utils.data import DataLoader
        try:
            from torch.utils.data import IterableDataset
            not_iterable = not isinstance(loader.dataset, IterableDataset)
        except Exception as e:
            not_iterable = LifeCycle
        return (isinstance(loader, DataLoader)
                and not_iterable)

    @abstractmethod
    def shutdown(self):
        """Attempts to shut down the worker."""
        pass

    # Internal variables must be accessible
    @property
    @abstractmethod
    def rank(self):
        pass

    @rank.setter
    @abstractmethod
    def rank(self, rank):
        pass

    @property
    @abstractmethod
    def backend(self):
        pass

    @backend.setter
    @abstractmethod
    def backend(self, backend):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @size.setter
    @abstractmethod
    def size(self, size):
        pass
