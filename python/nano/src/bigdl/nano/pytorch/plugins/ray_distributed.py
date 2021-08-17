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

# This file is adapted from ray_lightning. https://github.com/ray-project/ray_lightning
# Copyright 2021 Ray Lightning Author
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


from typing import Callable, Dict, List, Union, Any, Optional

import os
from collections import defaultdict

import ray
import torch
import intel_pytorch_extension as ipex
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning import _logger as log, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed
from ray.util.sgd.utils import find_free_port
from torch.nn import Module

from .environments.ray_envbase import RayEnvironment


@ray.remote
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key: str, value: str):
        """Set an environment variable with the provided values."""
        if value is not None:
            value = str(value)
            os.environ[key] = value

    def set_env_vars(self, keys: List[str], values: List[str]):
        """Sets multiple env vars with the provided values"""
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            self.set_env_var(key, value)

    def get_env_vars(self, key: str):
        return os.environ[key]

    def get_node_ip(self):
        """Returns the IP address of the node that this Ray actor is on."""
        return ray.services.get_node_ip_address()

    def execute(self, fn: Callable, *args, **kwargs):
        """Execute the provided function and return the result."""
        return fn(*args, **kwargs)


class RayPlugin(DDPSpawnPlugin):
    """Pytorch Lightning plugin for DDP training on a Ray cluster.
    This plugin is used to manage distributed training using DDP and
    Ray for process launching. Internally, the specified number of
    Ray actors are launched in the cluster and are registered as part of a
    Pytorch DDP process group. The Pytorch Lightning trainer is instantiated
    on the driver and sent to each of these training workers where training is
    executed. The distributed training protocol is handled by Pytorch DDP.
    Each training worker is configured to reserve ``num_cpus_per_worker``
    CPUS and 1 GPU if ``use_gpu`` is set to ``True``.
    If using this plugin, you should run your code like a normal Python
    script: ``python train.py``, and only on the head node if running in a
    distributed Ray cluster. There is no need to run this script on every
    single node.
    Args:
        num_workers (int): Number of training workers to use.
        num_cpus_per_worker (int): Number of CPUs per worker.
        use_gpu (bool): Whether to use GPU for allocation. For GPU to be
            used, you must also set the ``gpus`` arg in your Pytorch Lightning
            Trainer to a value > 0.
        init_hook (Callable): A function to run on each worker
            upon instantiation.
        **ddp_kwargs: Additional arguments to pass into
            ``DistributedDataParallel`` initialization
    Example:
        .. code_block:: python
            import pytorch_lightning as ptl
            from ray_lightning import RayAccelerator
            ptl_model = MNISTClassifier(...)
            plugin = RayPlugin(num_workers=4, cpus_per_worker=1,
                use_gpu=True)
            # If using GPUs, set the ``gpus`` arg to a value > 0.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., gpus=1, plugins=[plugin])
            trainer.fit(ptl_model)
    """

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 use_ipex: bool = False,
                 init_hook: Callable = None,
                 **ddp_kwargs: Union[Any, Dict[str, Any]]):

        # Unset MKL setting as bigdl.nano would give default values when init env.
        # Running different programs may need different configurations.
        # refer to https://analytics-zoo-doc.readthedocs.io/en/master/zoo.ray.html#
        # zoo.ray.raycontext.RayServiceFuncGenerator
        # Also KMP_AFFINITY may entangle something doing with ray._raylet.CoreWorker,
        # which is invoked by ray.init and leads to stocking.
        os.environ.pop("KMP_AFFINITY", None)
        os.environ.pop("OMP_NUM_THREADS", None)

        if not ray.is_initialized():
            print(ray.init())

        super().__init__(
            sync_batchnorm=False,
            parallel_devices=[],
            cluster_environment=RayEnvironment(world_size=num_workers),
            **ddp_kwargs)  # type: ignore
        self.nickname = "ddp_ray"
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.use_ipex = use_ipex

        assert not self.use_gpu or not self.use_ipex, \
            "You can not specify gpu and ipex at the same time."

        self.workers: List[Any] = []
        self.init_hook = init_hook
        self._local_rank = 0

    def _create_worker(self):
        """Creates Ray actor."""
        from bigdl.nano.common.cpu_schedule import schedule_workers

        cpu_procs = schedule_workers(self.num_workers)

        workers = []
        for i in range(self.num_workers):

            worker = RayExecutor.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=int(self.use_gpu)
            ).remote()

            KMP_AFFINITY_vars = f"granularity=fine,proclist"\
                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit"
            ray.get(worker.set_env_var.remote("KMP_AFFINITY", KMP_AFFINITY_vars))
            ray.get(worker.set_env_var.remote("OMP_NUM_THREADS", str(len(cpu_procs[i]))))

            workers.append(worker)

        return workers

    def setup(self, model):
        """Sets up PTL Trainer and creates the Ray actors."""
        # Check that trainer attribute has been set when this method is called.
        self._model = model
        self.workers = self._create_worker()
        if self.init_hook:
            ray.get([w.execute.remote(self.init_hook) for w in self.workers])

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["workers"]
        return d

    def __setstate__(self, d):
        d["workers"] = []
        self.__dict__.update(d)

    def get_local_ranks(self):
        """Creates a mapping of global ranks to local ranks."""
        # Get the local ranks for all the workers and store as a dict.
        # First get the IP address of each remote worker.
        node_ips = ray.get([w.get_node_ip.remote() for w in self.workers])
        rank_counter_dict = defaultdict(int)  # type: ignore
        global_to_local = [None] * self.num_workers
        for global_rank in range(self.num_workers):
            ip = node_ips[global_rank]
            global_to_local[global_rank] = rank_counter_dict[ip]
            rank_counter_dict[ip] += 1
        return global_to_local

    def _setup_env_vars(self):
        # Get rank 0 worker address and port for DDP connection.
        os.environ["MASTER_ADDR"] = ray.get(
            self.workers[0].get_node_ip.remote())
        os.environ["MASTER_PORT"] = str(
            ray.get(self.workers[0].execute.remote(find_free_port)))

        # Set environment variables for remote workers.
        keys = [
            "PL_GLOBAL_SEED", "PL_TORCH_DISTRIBUTED_BACKEND", "MASTER_ADDR",
            "MASTER_PORT"
        ]
        values = [os.getenv(k) for k in keys]
        ray.get([w.set_env_vars.remote(keys, values) for w in self.workers])

    def execution_loop(self, trainer, tune_enabled: bool = True):
        """Main execution loop for training, testing, & prediction.
        Sets up the torch.distributed process group for each
        worker. Then trigger remote training/testing/eval via
        ``train_remote`` on each worker. If using with Ray Tune, create a
        communication queue to retrieve intermediate results, and process
        those results. Finally retrieve the training results from the rank 0
        worker and return."""

        # Sets environment variables for all workers.
        self._setup_env_vars()

        self.global_to_local = self.get_local_ranks()

        model = self._model  # type: ignore
        model_ref = ray.put(model)
        # Don't pickle the model when training remotely.

        self._model = None  # type: ignore

        futures = [
            self.workers[i].execute.remote(self.execute_remote, model_ref, i)
            for i in range(self.num_workers)
        ]

        not_ready = futures
        while not_ready:
            ready, not_ready = ray.wait(not_ready, timeout=0)
            ray.get(ready)
        ray.get(ready)

        results = ray.get(futures)

        # Get the results, checkpoint path, and model weights from worker 0.
        results, best_path, state_dict = results[0]
        # Set the state for PTL using the output from remote training.
        self._results = results

        self._model = model  # type: ignore
        self._model.load_state_dict(state_dict)  # type: ignore
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback \
                .best_model_path = best_path

        return results

    def start_training(self, trainer):
        results = self.execution_loop(trainer, tune_enabled=True)
        # reset optimizers, since main process is never used for training and
        # thus does not have a valid optim state.
        trainer.optimizers = []
        return results

    def start_testing(self, trainer):
        results = self.execution_loop(trainer, tune_enabled=False)
        return results

    def start_predicting(self, trainer):
        results = self.execution_loop(trainer, tune_enabled=False)
        return results

    def post_dispatch(self):
        """Shutdown the DDP process group and all the Ray actors. """

        def shutdown_remote():
            torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ray.get([w.execute.remote(shutdown_remote) for w in self.workers])
        for w in self.workers:
            ray.kill(w, no_restart=True)
            del w
        self.workers = []

    # All methods below are only executed in remote Ray workers.

    def execute_remote(self,
                       model: LightningModule,
                       global_rank: int
                       ):
        """Train/test/eval function to be executed on each remote worker."""
        assert isinstance(self, RayPlugin)
        # This method should be executed remotely in each worker.
        self._model = model  # type: ignore
        self.lightning_module.trainer.accelerator_connector\
            ._training_type_plugin = self
        self.lightning_module.trainer.accelerator.training_type_plugin = self

        assert isinstance(self.cluster_environment, RayEnvironment)
        self.cluster_environment.set_global_rank(global_rank)
        self.cluster_environment.set_remote_execution(True)

        # Calling new_process will call
        # transfer_distrib_spawn_state_on_fit_end.
        # We override that method and have it just set attributes.
        # Then we can just return those attributes here.
        self.new_process(
            process_idx=global_rank,
            trainer=self.lightning_module.trainer,
            mp_queue=None)
        # Only need results from worker 0.
        if self.global_rank == 0:
            return self.results, self.best_model_path, self.model_state_dict
        else:
            return None

    def init_ddp_connection(self,  # type: ignore
                            global_rank: int,
                            world_size: int,
                            is_slurm_managing_tasks: bool = False) -> None:  # type: ignore
        """Process group creation to be executed on each remote worker."""
        torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
        if torch_backend is None:
            torch_backend = "nccl" if self.use_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER:"
                     f" {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                backend=torch_backend,
                rank=global_rank,
                world_size=world_size,
            )

    def set_world_ranks(self, process_idx: int = 0):
        """Set the appropriate rank attribues for the trainer."""
        assert self.cluster_environment is not None and \
            isinstance(self.cluster_environment, RayEnvironment)
        if self.cluster_environment.is_remote():
            self._local_rank = self.global_to_local[self.global_rank]
            self.cluster_environment.set_global_rank(self.global_rank)
            self.cluster_environment.set_world_size(self.num_workers)
            rank_zero_only.rank = self.cluster_environment.global_rank()  # type: ignore

    def new_process(self, process_idx, trainer, mp_queue):
        self.mp_queue = mp_queue

        reset_seed()

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank  # type: ignore

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: we moved it to the trainer.fit after calling pre_dispatch
        #   ... need to double check that it is the correct place
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with"
                     "{self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        self.configure_ddp()

        # Move this line here so that we can temporarily use cpu while configuring ddp
        # and use ipex.DEVICE later on
        # move the model to the correct device
        #
        # The reason for this movement is relate to unstorage tensor for ipex.
        # So maybe another way is replacing torch.save like ipexaccelerator does.
        self.model_to_device()

        self.barrier()
        results = trainer.run_stage()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(results)

    @property
    def root_device(self):
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda", 0)
        elif self.use_ipex:
            # Add ipex option.
            return torch.device(ipex.DEVICE)
        else:
            return torch.device("cpu")

    def determine_ddp_device_ids(self):
        # For ipex case, we also should not return any optional device id.
        if self.root_device.type == "cpu" or self.root_device.type == "xpu":
            return None
        return [self.root_device.index]

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        """Sets the training output as attributes so it can be retrieved."""
        if self.global_rank == 0:
            # Save training results as attributes.
            self._results = results

            # unsupported Storage type for ipex
            # Convert xpu tensor back to cpu
            # refer to https://github.com/intel/intel-extension-for-pytorch/issues/158
            if self.use_ipex:
                self.lightning_module.to("cpu")

            self.model_state_dict = self.lightning_module.state_dict()
            best_model_path = None
            if self.lightning_module.trainer.checkpoint_callback is not None:
                best_model_path = \
                    self.lightning_module.trainer.checkpoint_callback\
                        .best_model_path
            self.best_model_path = best_model_path

    @property
    def distributed_sampler_kwargs(self):
        """Returns the args to use for torch.data.DistributedSampler."""
        distributed_sampler_kwargs = dict(
            num_replicas=self.num_workers, rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def require_distributed_sampler(self):
        """This plugin requires a distributed sampler."""
        return True

    @property
    def is_distributed(self):
        return True
