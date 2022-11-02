from abc import abstractmethod, ABCMeta
import ray

from bigdl.orca.learn.pytorch.utils import find_free_port
from bigdl.dllib.utils.log4Error import *

from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


class LifeCycleManager(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.backend = "torch-distributed"
        self.rank = -1
        self.size = -1

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
        self._init_torchDDP(tcp_store_host, tcp_store_port, world_rank,
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

    def _init_torchDDP(self, tcp_store_host, tcp_store_port, world_rank,
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
            not_iterable = LifeCycleManager
        return (isinstance(loader, DataLoader)
                and not_iterable)

    @abstractmethod
    def shutdown(self):
        """Attempts to shut down the worker."""
        pass
