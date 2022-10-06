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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/torch_runner.py

from filelock import FileLock
import logging
import io
import itertools
import os
import copy
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from bigdl.orca import OrcaContext
from bigdl.orca.learn.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from bigdl.orca.learn.pytorch.constants import SCHEDULER_STEP, NUM_STEPS
from bigdl.orca.learn.pytorch.training_operator import TrainingOperator
from bigdl.orca.learn.pytorch import utils
from bigdl.orca.learn.pytorch.utils import get_filesystem
from bigdl.dllib.utils.log4Error import *

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class DistBackend:

    def get_world_size(self):
        pass

    def all_reduce(self, *args, **kwargs):
        pass


class HorovodDistBackend(DistBackend):

    def get_world_size(self):
        import horovod.torch as hvd
        return hvd.size()

    def all_reduce(self, *args, **kwargs):
        import horovod.torch as hvd
        return hvd.allreduce(*args, **kwargs)


class TorchDistBackend(DistBackend):

    def get_world_size(self):
        import torch.distributed as dist
        return dist.get_world_size()

    def all_reduce(self, *args, **kwargs):
        import torch.distributed as dist
        return dist.all_reduce(*args, **kwargs)

    def is_initialized(self):
        import torch.distributed as dist
        return dist.is_initialized()

    def all_reduce_min(self, tensor, *args, **kwargs):
        import torch.distributed as dist
        all_reduce_min_kwargs = dict(op=dist.ReduceOp.MIN)
        all_reduce_min_kwargs.update(kwargs)
        return dist.all_reduce(tensor, *args,
                               **all_reduce_min_kwargs)


class TorchRunner:
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 training_operator_cls=None,
                 config=None,
                 use_tqdm=False,
                 scheduler_step_freq=None,
                 sync_stats=True,
                 log_level=logging.INFO):
        logging.basicConfig(level=log_level,
                            format='[%(asctime)s] %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
        self.logger = logging.getLogger(__name__)
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls or TrainingOperator
        self.config = {} if config is None else config

        self.timers = utils.TimerCollection()
        self.epochs = 0
        self.models = None
        self.optimizers = None
        self.metrics = metrics
        self.criterion = None
        self.schedulers = None
        self.train_loader = None
        self.validation_loader = None
        self.training_operator = None
        self.use_tqdm = use_tqdm
        self.scheduler_step_freq = scheduler_step_freq
        self.sync_stats = sync_stats
        self.epochs_stats = None  # The state saved in every epoch

    def _create_loss(self):
        if not self.loss_creator:
            return
        self.logger.debug("Creating loss.")
        if isinstance(self.loss_creator, torch.nn.modules.loss._Loss):
            self.criterion = self.loss_creator
        else:  # Torch loss is also callable.
            import types
            invalidInputError(isinstance(self.loss_creator, types.FunctionType),
                              "Must provide a torch loss instance or a loss_creator function")
            self.criterion = self.loss_creator(self.config)

    def _create_schedulers_if_available(self):
        # Learning rate schedules are optional.
        if not self.scheduler_creator:
            return
        self.schedulers = self.scheduler_creator(self.given_optimizers,
                                                 self.config)

        if not isinstance(self.schedulers, Iterable):
            self.schedulers = [self.schedulers]

    def setup(self, cores_per_node):
        import torch
        torch.set_num_threads(cores_per_node)

    def setup_torch_distribute(self, tcp_store_host, tcp_store_port, world_rank,
                               world_size):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        client_store = dist.TCPStore(tcp_store_host, tcp_store_port, -1, False)
        dist.init_process_group(
            backend="gloo",
            store=client_store,
            rank=world_rank,
            world_size=world_size)
        self.backend = "torch-distributed"
        self.rank = world_rank
        self.size = world_size
        self.setup_components()
        training_models = [
            DistributedDataParallel(model)
            for model in self.models
        ]
        self.setup_operator(training_models)

    def setup_components(self):
        """Runs the creator functions without any distributed coordination."""

        self.logger.debug("Creating model")
        self.models = self.model_creator(self.config)
        if isinstance(self.models, nn.Sequential) or not isinstance(self.models, Iterable):
            self.models = [self.models]
        invalidInputError(all(isinstance(model, nn.Module) for model in self.models),
                          ("All models must be PyTorch models: {}.".format(self.models)))

        self.logger.debug("Creating optimizer.")
        self.optimizers = self.optimizer_creator(self.given_models,
                                                 self.config)
        if not isinstance(self.optimizers, Iterable):
            self.optimizers = [self.optimizers]

        self._create_schedulers_if_available()
        self._create_loss()

    def setup_operator(self, training_models):
        """Create the training operator."""
        if self.backend == "horovod":
            self.dist_backend = HorovodDistBackend()
        else:
            self.dist_backend = TorchDistBackend()

        self.training_operator = \
            self.training_operator_cls(
                self.config,
                models=training_models,
                optimizers=self.optimizers,
                criterion=self.criterion,
                world_rank=self.rank,
                schedulers=self.schedulers,
                use_tqdm=self.use_tqdm,
                sync_stats=self.sync_stats,
                dist_backend=self.dist_backend)

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
            not_iterable = TorchRunner
        return (isinstance(loader, DataLoader)
                and not_iterable)

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False,
                     info=None, wrap_dataloader=None, callbacks=None,
                     validation_data_creator=None):
        config = copy.copy(self.config)
        if OrcaContext.serialize_data_creator:
            with FileLock(
                    os.path.join(tempfile.gettempdir(), ".orcadata.lock")):
                loader = data_creator(config, batch_size)
        else:
            loader = data_creator(config, batch_size)

        if wrap_dataloader is None:
            if TorchRunner.should_wrap_dataloader(loader):
                loader = self.with_sampler(loader)
        elif wrap_dataloader is True:
            loader = self.with_sampler(loader)

        if validation_data_creator:
            if OrcaContext.serialize_data_creator:
                with FileLock(
                        os.path.join(tempfile.gettempdir(), ".orca_val_data.lock")):
                    val_loader = validation_data_creator(config, batch_size)
            else:
                val_loader = validation_data_creator(config, batch_size)

            wrapped = False
            if wrap_dataloader is None:
                if TorchRunner.should_wrap_dataloader(val_loader):
                    val_loader = self.with_sampler(val_loader)
                    wrapped = True
            elif wrap_dataloader is True:
                val_loader = self.with_sampler(val_loader)
                wrapped = True

            if not wrapped:
                # Truncate validation by the min step for all workers (data may distribute unevenly)
                # Or it results in error in next epoch of training (op.preamble.length <= op.nbytes)
                validation_tensor = torch.tensor(len(val_loader))
                invalidInputError(self.backend != "horovod", "Sanity check failed!")
                self.dist_backend.all_reduce_min(validation_tensor)
                val_steps = validation_tensor.item()
            else:
                val_steps = None

        else:
            val_loader = None
            val_steps = None

        if callbacks is not None:
            for callback in callbacks:
                callback.set_model(self.given_models)
                if hasattr(callback, "set_trainer"):
                    callback.set_trainer(self)
                callback.on_train_begin()
        stats_list = list()
        for i in range(epochs):
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_begin(epoch=self.epochs)
            stats = self.train_epoch(loader, profile=profile, info=info, callbacks=callbacks,
                                     val_loader=val_loader, val_steps=val_steps)
            if self.rank == 0:
                if self.sync_stats:
                    self.logger.info(f"Finished training epoch {i + 1}, " +
                                     f"stats averaged over workers: {stats}")
                else:
                    self.logger.info(f"Finished training epoch {i + 1}, " +
                                     f"stats on rank 0: {stats}")
            stats_list.append(stats)
            self.epochs_stats = stats
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch=self.epochs, logs=self.epochs_stats)
        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_end(logs=self.epochs_stats)
        return stats_list

    def train_epoch(self,
                    data_loader,
                    profile=False,
                    info=None,
                    callbacks=None,
                    val_loader=None,
                    val_steps=None):
        """Runs a training epoch and updates the model parameters."""
        if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.epochs)
        self.logger.debug("Begin Training Step {}".format(self.epochs + 1))

        info = info or {}
        self._toggle_profiling(profile=profile)

        info.update({
            SCHEDULER_STEP: self.scheduler_step_freq
        })
        with self.timers.record("train_epoch"):
            data_loader = iter(data_loader)
            train_stats = self.training_operator.train_epoch(data_loader, info, callbacks)

        if val_loader:
            with self.timers.record("validation"):
                info = info or {}
                validation_results = self.training_operator.validate(val_loader,
                                                                     info=info,
                                                                     metrics=self.metrics,
                                                                     num_steps=val_steps)
                # add prefix of "val_" for validation_stats
                validation_stats = {}
                for name, value in validation_results.items():
                    if not name.startswith("val_"):
                        name = "val_" + name.lower()
                    validation_stats[name] = value

        else:
            validation_stats = {}

        self.epochs += 1
        # This is so that `epochs` is first in ordering.
        stats = dict(epoch=self.epochs, **train_stats, **validation_stats)

        if profile:
            stats.update(profile=self.timers.stats())

        return stats

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 info=None, wrap_dataloader=None):
        """Evaluates the model on the validation data set."""
        config = copy.copy(self.config)
        info = info or {}
        self._toggle_profiling(profile=profile)

        if OrcaContext.serialize_data_creator:
            with FileLock(
                    os.path.join(tempfile.gettempdir(), ".orcadata.lock")):
                loader = data_creator(config, batch_size)
        else:
            loader = data_creator(config, batch_size)

        if wrap_dataloader is None:
            if TorchRunner.should_wrap_dataloader(loader):
                loader = self.with_sampler(loader)
        elif wrap_dataloader is True:
            loader = self.with_sampler(loader)
        loader = iter(loader)
        if num_steps:
            loader = itertools.islice(loader, num_steps)
        with self.timers.record("validation"):
            validation_stats = self.training_operator.validate(loader,
                                                               info=info,
                                                               metrics=self.metrics)
        if profile:
            validation_stats.update(profile=self.timers.stats())
        return validation_stats

    def predict(self, partition, batch_size=32, profile=False):
        """Evaluates the model on the validation data set."""
        config = copy.copy(self.config)
        self._toggle_profiling(profile=profile)

        params = {"batch_size": batch_size, "shuffle": False}
        for arg in ["shuffle", "sampler", "batch_sampler", "num_workers", "collate_fn",
                    "pin_memory", "drop_last", "timeout", "worker_init_fn",
                    "multiprocessing_context"]:
            if arg in config:
                params[arg] = config[arg]

        def predict_fn(shard):
            if isinstance(partition, IterableDataset):
                y = self.training_operator.predict(shard)
            else:
                if isinstance(shard["x"], tuple) or isinstance(shard["x"], list):
                    tensors = [torch.from_numpy(arr) for arr in shard["x"]]
                else:
                    tensors = [torch.from_numpy(shard["x"])]
                dataset = torch.utils.data.TensorDataset(*tensors)
                data_loader = DataLoader(dataset, **params)
                y = self.training_operator.predict(iter(data_loader))
            return {"prediction": y}

        with self.timers.record("predict"):
            if isinstance(partition, IterableDataset):
                new_part = [predict_fn(shard) for shard, shard_idx in partition]
            else:
                new_part = [predict_fn(shard) for shard in partition]
        return new_part

    def _toggle_profiling(self, profile=False):
        """Enables/Disables and resets timing profiles."""
        if profile:
            self.timers.enable()
            self.timers.reset()
        else:
            self.timers.disable()
        self.training_operator._set_timers(self.timers)

    def get_state_dict(self):
        """Returns the state of the runner."""
        state = {
            "epoch": self.epochs,
            "operator": self.training_operator.state_dict(),
            "models": [model.state_dict() for model in self.models],
            "optimizers": [opt.state_dict() for opt in self.optimizers]
        }
        if self.schedulers:
            state.update({
                "schedulers": [
                    scheduler.state_dict() for scheduler in self.schedulers
                ]
            })
        return state

    def load_state_dict(self, state):
        """Sets the state of the model."""
        for model, state_dict in zip(self.models, state["models"]):
            model.load_state_dict(state_dict)
        if "optimizers" in state:
            for optimizer, state_dict in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(state_dict)
        if self.schedulers and "schedulers" in state:
            for scheduler, state_dict in zip(self.schedulers,
                                             state["schedulers"]):
                scheduler.load_state_dict(state_dict)

        self.epochs = state["epoch"]
        if "operator" in state:
            self.training_operator.load_state_dict(state["operator"])

    @staticmethod
    def _state_dict2stream(state_dict):
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        return _buffer.getvalue()

    @staticmethod
    def _state_stream2dict(byte_obj):
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer)
        return state_dict

    def get_state_stream(self):
        """Returns a bytes object for the state dict."""
        state_dict = self.get_state_dict()
        state_stream = TorchRunner._state_dict2stream(state_dict)
        return state_stream

    def load_state_stream(self, byte_obj):
        """Loads a bytes object the training state dict."""
        state_dict = TorchRunner._state_stream2dict(byte_obj)
        return self.load_state_dict(state_dict)

    def save_checkpoint(self, filepath, save_weights_only=False):
        if self.rank == 0:
            self._save_checkpoint(filepath, save_weights_only)
            self.logger.debug(f"Saved checkpoint: {filepath}")
        return filepath

    def _save_checkpoint(self, filepath, save_weights_only=False):
        import fsspec
        if save_weights_only:
            checkpoint = {
                "epoch": self.epochs,
                "models": [model.state_dict() for model in self.models],
            }
        else:
            checkpoint = self.get_state_dict()
        byte_obj = TorchRunner._state_dict2stream(checkpoint)
        with fsspec.open(filepath, "wb") as f:
            f.write(byte_obj)

    def load_checkpoint(self, filepath):
        fs = get_filesystem(filepath)
        if not fs.exists(filepath):
            invalidInputError(False,
                              f"Checkpoint at {filepath} not found. Aborting training.")
        with fs.open(filepath, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def remove_checkpoint(self, filepath):
        if self.rank == 0:
            self._remove_checkpoint(filepath)

    def _remove_checkpoint(self, filepath):
        fs = get_filesystem(filepath)
        if fs.exists(filepath):
            fs.rm(filepath, recursive=True)
            self.logger.debug(f"Removed checkpoint: {filepath}")

    def apply(self, fn):
        return fn()

    def apply_operator(self, fn):
        return fn(self.training_operator)

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.training_operator
        del self.validation_loader
        del self.train_loader
        del self.criterion
        del self.optimizers
        del self.models

    @property
    def given_models(self):
        if len(self.models) > 1:
            return self.models
        else:
            return self.models[0]

    @property
    def given_optimizers(self):
        if len(self.optimizers) > 1:
            return self.optimizers
        else:
            return self.optimizers[0]

    @property
    def given_schedulers(self):
        if not self.schedulers:
            return self.schedulers
        if len(self.schedulers) > 1:
            return self.schedulers
        else:
            return self.schedulers[0]
