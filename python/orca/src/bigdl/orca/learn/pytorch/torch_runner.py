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
import os
import copy
import tempfile
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from bigdl.orca.learn.metrics import Metric
from torch.utils.data.distributed import DistributedSampler
from bigdl.orca import OrcaContext
from bigdl.orca.learn.pytorch.constants import (SCHEDULER_STEP, SCHEDULER_STEP_EPOCH,
                                                SCHEDULER_STEP_BATCH)
from bigdl.orca.learn.pytorch import utils
from bigdl.orca.learn.pytorch.utils import (get_filesystem, AverageMeterCollection,
                                            NUM_SAMPLES, get_batchsize)
from bigdl.orca.learn.pytorch.core import BaseRunner
from bigdl.dllib.utils.log4Error import invalidInputError

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


class TorchRunner(BaseRunner):
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 config=None,
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
        self.config = {} if config is None else config

        self.timers = utils.TimerCollection()
        self.epochs = 0
        self.global_step = 0
        self.models = None
        self.optimizers = None
        self.metrics = metrics
        self.criterion = None
        self.schedulers = None
        self.train_loader = None
        self.validation_loader = None
        self.sync_stats = sync_stats
        self.epoch_stats = None  # The state saved in every epoch
        self._mode = 'val'  # By default we don't use ddp model

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

    def setup_components(self):
        """Runs the creator functions without any distributed coordination."""

        self.logger.debug("Creating model")
        if self.model_creator:
            self.models = self.model_creator(self.config)

            if isinstance(self.models, nn.Sequential) or not isinstance(self.models, Iterable):
                self.models = [self.models]
            invalidInputError(all(isinstance(model, nn.Module) for model in self.models),
                                 ("All models must be PyTorch models: {}.".format(self.models)))

            if self.optimizer_creator:
                self.logger.debug("Creating optimizer.")
                self.optimizers = self.optimizer_creator(self.given_models,
                                                         self.config)
                if self.optimizers and not isinstance(self.optimizers, Iterable):
                    self.optimizers = [self.optimizers]

        self._create_schedulers_if_available()
        self._create_loss()

    def setup_ddp_components(self):
        from torch.nn.parallel import DistributedDataParallel
        self.training_models = [
            DistributedDataParallel(model)
            for model in self.models
        ]
        self.setup_operator(self.training_models)

    def setup_operator(self, training_models):
        """Create the training operator."""
        if self.backend == "horovod":
            self.dist_backend = HorovodDistBackend()
        else:
            self.dist_backend = TorchDistBackend()

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False,
                     wrap_dataloader=None, callbacks=None,
                     validation_data_creator=None):
        config = copy.copy(self.config)
        self.num_epochs = epochs
        if OrcaContext.serialize_data_creator:
            with FileLock(
                    os.path.join(tempfile.gettempdir(), ".orcadata.lock")):
                self.train_loader = data_creator(config, batch_size)
        else:
            self.train_loader = data_creator(config, batch_size)

        if wrap_dataloader is None:
            if TorchRunner.should_wrap_dataloader(self.train_loader):
                self.train_loader = self.with_sampler(self.train_loader)
        elif wrap_dataloader is True:
            self.train_loader = self.with_sampler(self.train_loader)

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

        self.call_hook(callbacks=callbacks, fn_name="before_run")

        stats_list = list()
        for i in range(epochs):
            del self.epoch_stats
            self.call_hook(callbacks=callbacks, fn_name="before_train_epoch")
            stats = self.train_epoch(self.train_loader, profile=profile,
                                     callbacks=callbacks, val_loader=val_loader,
                                     val_steps=val_steps)
            self.epoch_stats = stats
            self.call_hook(callbacks=callbacks, fn_name="after_train_epoch")

            if self.rank == 0:
                if self.sync_stats:
                    self.logger.info(f"Finished training epoch {i + 1}, " +
                                     f"stats averaged over workers: {stats}")
                else:
                    self.logger.info(f"Finished training epoch {i + 1}, " +
                                     f"stats on rank 0: {stats}")
            stats_list.append(stats)

        self.call_hook(callbacks=callbacks, fn_name="after_run")

        return stats_list

    def train_epoch(self,
                    data_loader,
                    profile=False,
                    callbacks=None,
                    val_loader=None,
                    val_steps=None):
        """Runs a training epoch and updates the model parameters."""
        if hasattr(data_loader, "sampler") and hasattr(
                data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(self.epochs)
        self.logger.debug("Begin Training Step {}".format(self.epochs + 1))

        if not self.criterion:
            invalidInputError(False,
                              "You must provide a loss for train and evaluate.")

        self._toggle_profiling(profile=profile)

        with self.timers.record("train_epoch"):
            data_loader = iter(data_loader)
            train_stats = self._train_epoch(data_loader, callbacks)

        if val_loader:
            with self.timers.record("validation"):
                validation_results = self._validate(val_loader,
                                                    metrics=self.metrics,
                                                    num_steps=val_steps,
                                                    callbacks=callbacks)
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

    def _train_epoch(self, iterator, callbacks=None):
        """Runs one standard training pass over the training dataloader.

        By default, this method will iterate over the given iterator and
        call ``self.train_batch`` over each batch.

        You do not need to call ``train_batch`` in this method if you plan
        to implement a custom optimization/training routine here.

        You may find ``ray.util.sgd.utils.AverageMeterCollection`` useful
        when overriding this method. See example below:

        .. code-block:: python

            def train_epoch(self, ...):
                meter_collection = AverageMeterCollection()
                self.model.train()
                for batch in iterator:
                    # do some processing
                    metrics = {"metric_1": 1, "metric_2": 3} # dict of metrics

                    # This keeps track of all metrics across multiple batches
                    meter_collection.update(metrics, n=len(batch))

                # Returns stats of the meters.
                stats = meter_collection.summary()
                return stats


        Args:
            iterator (iter): Iterator over the training data for the entire
                epoch. This iterator is expected to be entirely consumed.

        Returns:
            A dict of metrics from training.
        """
        self._mode = 'train'

        metric_meters = AverageMeterCollection()

        # TODO: Discuss the situation when there are multiple components,
        #       It is best for the user to write this part of the logic in a hook func.
        self.model.train()
        # self.training_models may not be DDP if horovod.
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(self.model, DDP):
            with self.model.join():
                self._train_loop(iterator, metric_meters, callbacks)
        else:
            self._train_loop(iterator, metric_meters, callbacks)

        self.call_hook(callbacks=callbacks, fn_name="on_lr_adjust")

        return metric_meters.summary(sync_stats=self.sync_stats,
                                     dist_backend=self.dist_backend)

    def _train_loop(self, iterator, metric_meters, callbacks):
        for batch_idx, batch in enumerate(iterator):
            self.batch_idx = batch_idx

            self._train_batch(batch, callbacks=callbacks)

            metric_meters.update(self.metrics_stats, n=self.metrics_stats.pop(NUM_SAMPLES, 1))
            self.global_step += 1

            del self.batch_idx
            del self.metrics_stats

    def _train_batch(self, batch, callbacks=None):
        """Computes loss and updates the model over one batch.

        This method is responsible for computing the loss and gradient and
        updating the model.

        By default, this method implementation assumes that batches
        are in (\*features, labels) format. So we also support multiple inputs
        model. If using amp/fp16 training, it will also scale the loss
        automatically.

        You can provide custom loss metrics and training operations if you
        override this method. If overriding this method, you can access model,
        optimizer, criterion via ``self.model``, ``self.optimizer``,
        and ``self.criterion``.

        You do not need to override this method if you plan to
        override ``train_epoch``.

        Args:
            batch: One item of the validation iterator.

        Returns:
            A dictionary of metrics.
                By default, this dictionary contains "loss" and "num_samples".
                "num_samples" corresponds to number of datapoints in the batch.
                However, you can provide any number of other values.
                Consider returning "num_samples" in the metrics because
                by default, ``train_epoch`` uses "num_samples" to
                calculate averages.

        """
        # unpack features into list to support multiple inputs model
        # and restore batch to what it should be.
        features, target = batch
        if torch.is_tensor(features):
            self.batch = features, target
        elif isinstance(features, (tuple, list)):
            self.batch = *features, target
        else:
            invalidInputError(False,
                              "Features should be tensor or list/tuple, "
                              "but got {}".format(type(features)))
        self.call_hook(callbacks=callbacks, fn_name="before_train_iter")

        # Compute output.
        with self.timers.record("fwd"):
            self.call_hook(callbacks=callbacks, fn_name="on_train_forward")

        # Compute gradients in a backward pass.
        with self.timers.record("bwd"):
            self.call_hook(callbacks=callbacks, fn_name="on_iter_backward")

        loss_item = self.loss.item()
        self.metrics_stats = {"train_loss": loss_item, NUM_SAMPLES: get_batchsize(features)}
        self.call_hook(callbacks=callbacks, fn_name="after_train_iter")

        # User should not see batch/loss from last iteration
        del self.batch
        del self.output
        del self.loss

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 wrap_dataloader=None, callbacks=None):
        """Evaluates the model on the validation data set."""
        if not self.criterion:
            invalidInputError(False,
                              "You must provide a loss for train and evaluate.")

        config = copy.copy(self.config)
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

        with self.timers.record("validation"):
            validation_stats = self._validate(loader,
                                              metrics=self.metrics,
                                              num_steps=num_steps,
                                              callbacks=callbacks)
        if profile:
            validation_stats.update(profile=self.timers.stats())
        return validation_stats

    def _validate(self, val_iterator, metrics, num_steps=None, callbacks=None):
        """Runs one standard validation pass over the val_iterator.

        This will call ``model.eval()`` and ``torch.no_grad`` when iterating
        over the validation dataloader.

        If overriding this method, you can access model, criterion via
        ``self.model`` and ``self.criterion``. You also do not need to call
        ``validate_batch`` if overriding this method.

        Args:
            val_iterator (iter): Iterable constructed from the
                validation dataloader.

        Returns:
            A dict of metrics from the evaluation.
                By default, returns "val_accuracy" and "val_loss"
                which is computed by aggregating "loss" and "correct" values
                from ``validate_batch`` and dividing it by the sum of
                ``num_samples`` from all calls to ``self.validate_batch``.
        """
        # switch to evaluate mode
        self._mode = 'val'
        self.model.eval()
        metrics = Metric.convert_metrics_dict(metrics, backend="pytorch")
        losses = []
        total_samples = 0
        self.call_hook(callbacks=callbacks, fn_name="before_val_epoch")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                self.batch_idx = batch_idx
                if num_steps and batch_idx == num_steps:
                    break
                output, target, loss = self.forward_batch(batch, callbacks)
                num_samples = get_batchsize(target)
                total_samples += num_samples
                losses.append(loss.item() * num_samples)
                for metric in metrics.values():
                    metric(output, target)

                del self.batch_idx

        self.call_hook(callbacks=callbacks, fn_name="after_val_epoch")
        result = {name: metric.compute() for name, metric in metrics.items()}

        result["val_loss"] = sum(losses) / total_samples

        result["num_samples"] = total_samples

        return result

    def forward_batch(self, batch, callbacks=None):
        """Calculates the loss and accuracy over a given batch.

        You can override this method to provide arbitrary metrics.

        Same as ``train_batch``, this method implementation assumes that
        batches are in (\*features, labels) format by default. So we also
        support multiple inputs model.

        Args:
            batch: One item of the validation iterator.

        Returns:
            A dict of metrics.
                By default, returns "val_loss", "val_accuracy", and
                "num_samples". When overriding, consider returning
                "num_samples" in the metrics because
                by default, ``validate`` uses "num_samples" to
                calculate averages.
        """
        # unpack features into list to support multiple inputs model
        # and restore batch to what it should be.
        features, target = batch
        if torch.is_tensor(features):
            self.batch = features, target
        elif isinstance(features, (tuple, list)):
            self.batch = *features, target
        else:
            invalidInputError(False,
                              "Features should be tensor, list/tuple, "
                              "but got {}".format(type(features)))
        self.call_hook(callbacks=callbacks, fn_name="before_val_iter")

        # compute output
        with self.timers.record("eval_fwd"):
            self.call_hook(callbacks=callbacks, fn_name="on_val_forward")

        self.call_hook(callbacks=callbacks, fn_name="after_val_iter")

        # User should not see batch from last iteration
        output = self.output
        del self.batch
        del self.output

        return output, target, self.loss

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
                y = self._predict(shard)
            else:
                if isinstance(shard["x"], tuple) or isinstance(shard["x"], list):
                    tensors = [torch.from_numpy(arr) for arr in shard["x"]]
                else:
                    tensors = [torch.from_numpy(shard["x"])]
                dataset = torch.utils.data.TensorDataset(*tensors)
                data_loader = DataLoader(dataset, **params)
                y = self._predict(iter(data_loader))
            return {"prediction": y}

        with self.timers.record("predict"):
            if isinstance(partition, IterableDataset):
                new_part = [predict_fn(shard) for shard, shard_idx in partition]
            else:
                new_part = [predict_fn(shard) for shard in partition]
        return new_part

    def _predict(self, pred_iterator):
        # switch to evaluate mode
        self._mode = 'predict'
        self.model.eval()
        result = []
        with torch.no_grad():
            for batch in pred_iterator:
                result.append(self.predict_batch(batch))

        return np.concatenate(result, axis=0)

    def predict_batch(self, batch):

        if isinstance(batch, torch.Tensor):
            batch = [batch]

        # compute output
        with self.timers.record("pred_fwd"):
            output = self.model(*batch)

            if len(output.size()) > 1:
                # In case there is extra trailing dimensions.
                for i in reversed(range(1, len(output.size()))):
                    output = torch.squeeze(output, i)

        # todo support multi-output model
        np_output = output.detach().numpy()
        return np_output

    def _toggle_profiling(self, profile=False):
        """Enables/Disables and resets timing profiles."""
        if profile:
            self.timers.enable()
            self.timers.reset()
        else:
            self.timers.disable()

    def get_state_dict(self):
        """Returns the state of the runner."""
        state = {
            "epoch": self.epochs,
            "models": [model.state_dict() for model in self.models]
        }
        if self.optimizers:
            state.update({
                "optimizers": [
                    opt.state_dict() for opt in self.optimizers
                ]
            })
        if self.schedulers:
            state.update({
                "schedulers": [
                    scheduler.state_dict() for scheduler in self.schedulers
                ]
            })
        return state

    def load_state_dict(self, state):
        """Sets the state of the model."""
        import collections
        if isinstance(state, collections.OrderedDict):
            for model, state_dict in zip(self.models, [state]):
                model.load_state_dict(state_dict)
        else:
            if "models" in state:
                for model, state_dict in zip(self.models, state["models"]):
                    model.load_state_dict(state_dict)
            else:
                for model, state_dict in zip(self.models, state):
                    model.load_state_dict(state_dict)
        if self.optimizers and "optimizers" in state:
            for optimizer, state_dict in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(state_dict)
        if self.schedulers and "schedulers" in state:
            for scheduler, state_dict in zip(self.schedulers,
                                             state["schedulers"]):
                scheduler.load_state_dict(state_dict)
        if "epoch" in state:
            self.epochs = state["epoch"]

    def save_checkpoint(self, filepath, save_weights_only=False):
        if self.rank == 0:
            if save_weights_only:
                checkpoint = {
                    "epoch": self.epochs,
                    "models": [model.state_dict() for model in self.models],
                }
            else:
                checkpoint = self.get_state_dict()
            byte_obj = TorchRunner._state_dict2stream(checkpoint)
            file_name = os.path.basename(filepath)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_name)
            with open(temp_path, "wb") as f:
                f.write(byte_obj)
            from bigdl.orca.data.file import put_local_file_to_remote
            put_local_file_to_remote(temp_path, filepath)
            self.logger.debug(f"Saved checkpoint: {filepath}")
        return filepath

    def remove_checkpoint(self, filepath):
        if self.rank == 0:
            from bigdl.orca.data.file import exists, rmdir
            if exists(filepath):
                rmdir(filepath)
                self.logger.debug(f"Removed checkpoint: {filepath}")

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.validation_loader
        del self.train_loader
        del self.criterion
        del self.optimizers
        del self.models

    def call_hook(self, callbacks, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "on_iter_begin".
        """
        for hook in callbacks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

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
    def model(self):
        """
        First or only model(s) created by the ``model_creator``.
        Discuss whether to return ddp model depending on the mode.
        """
        if self._mode == 'train':
            if self.training_models:
                return self.training_models[0]
        else:
            if self.models:
                return self.models[0]

    @property
    def optimizer(self):
        """First or only optimizer(s) created by the ``optimizer_creator``."""
        return self.optimizers[0]

    @property
    def scheduler(self):
        """First or only scheduler(s) created by the ``scheduler_creator``."""
        if self.schedulers:
            return self.schedulers[0]
