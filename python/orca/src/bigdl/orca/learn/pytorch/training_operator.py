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
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/training_operator.py

import collections
import torch
import numpy as np

from bigdl.orca.learn.metrics import Metric
from bigdl.orca.learn.pytorch.utils import (TimerCollection, AverageMeterCollection,
                                            NUM_SAMPLES, get_batchsize)
from bigdl.orca.learn.pytorch.constants import (SCHEDULER_STEP_EPOCH, NUM_STEPS,
                                                SCHEDULER_STEP_BATCH, SCHEDULER_STEP)
from torch.nn.parallel import DistributedDataParallel as DDP
from bigdl.dllib.utils.log4Error import *


tqdm = None
try:
    from tqdm import tqdm
except ImportError:
    pass


def _is_multiple(component):
    """Checks if a component (optimizer, model, etc) is not singular."""
    return isinstance(component, collections.Iterable) and len(component) > 1


class TrainingOperator:
    """Abstract class for custom training or validation loops.

    The scheduler will only be called at a batch or epoch frequency, depending
    on the user parameter. Be sure to set ``scheduler_step_freq`` in
    ``TorchTrainer`` to either "batch" or "epoch" to increment the scheduler
    correctly during training. If using a learning rate scheduler
    that depends on validation loss, you can use ``trainer.update_scheduler``.

    For both training and validation, there are two granularities that
    you can provide customization: per epoch or per batch.
    You do not need to override both.

    .. image:: raysgd-custom.jpg
        :scale: 80%
        :align: center

    Raises:
        ValueError if multiple models/optimizers/schedulers are provided.
            You are expected to subclass this class if you wish
            to train over multiple models/optimizers/schedulers.
    """

    def __init__(self,
                 config,
                 models,
                 optimizers,
                 world_rank,
                 criterion=None,
                 schedulers=None,
                 use_fp16=False,
                 use_tqdm=False,
                 sync_stats=False,
                 dist_backend=None):
        # You are not expected to override this method.
        self._models = models  # List of models
        invalidInputError(isinstance(models, collections.Iterable),
                          "Components need to be iterable. Got: {}".format(type(models)))
        self._optimizers = optimizers  # List of optimizers
        if optimizers:
            invalidInputError(isinstance(optimizers, collections.Iterable),
                              "Components need to be iterable. Got: {}".format(type(optimizers)))
        self._world_rank = world_rank
        self._criterion = criterion
        self._schedulers = schedulers
        if schedulers:
            invalidInputError(isinstance(schedulers, collections.Iterable),
                              "Components need to be iterable. Got: {}".format(type(schedulers)))
        self._config = config
        self._use_fp16 = use_fp16
        if tqdm is None and use_tqdm:
            invalidInputError(False,
                              "tqdm must be installed to use tqdm in training.")
        self._use_tqdm = use_tqdm
        self.global_step = 0
        self.sync_stats = sync_stats
        self.dist_backend = dist_backend

        if type(self) is TrainingOperator:
            for component in (models, schedulers, optimizers):
                if _is_multiple(component):
                    invalidInputError(False,
                                      "Need to provide a custom operator subclassing "
                                      "TrainingOperator if using multi-scheduler, "
                                      "multi-model or multi-optimizer training/validation.")
        self.timers = TimerCollection()
        self.setup(config)

    def _set_timers(self, timers):
        """Passes in the timers from the Runner."""
        self.timers = timers

    def setup(self, config):
        """Override this method to implement custom operator setup.

        Args:
            config (dict): Custom configuration value to be passed to
                all creator and operator constructors. Same as ``self.config``.
        """
        pass

    def validate(self, val_iterator, info, metrics, num_steps=None):
        """Runs one standard validation pass over the val_iterator.

        This will call ``model.eval()`` and ``torch.no_grad`` when iterating
        over the validation dataloader.

        If overriding this method, you can access model, criterion via
        ``self.model`` and ``self.criterion``. You also do not need to call
        ``validate_batch`` if overriding this method.

        Args:
            val_iterator (iter): Iterable constructed from the
                validation dataloader.
            info: (dict): Dictionary for information to be used for custom
                validation operations.

        Returns:
            A dict of metrics from the evaluation.
                By default, returns "val_accuracy" and "val_loss"
                which is computed by aggregating "loss" and "correct" values
                from ``validate_batch`` and dividing it by the sum of
                ``num_samples`` from all calls to ``self.validate_batch``.
        """
        # switch to evaluate mode
        self.model.eval()
        metrics = Metric.convert_metrics_dict(metrics, backend="pytorch")
        losses = []
        total_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                if num_steps and batch_idx == num_steps:
                    break
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                output, target, loss = self.forward_batch(batch, batch_info)
                num_samples = get_batchsize(target)
                total_samples += num_samples
                losses.append(loss.item() * num_samples)
                for metric in metrics.values():
                    metric(output, target)

        result = {name: metric.compute() for name, metric in metrics.items()}

        result["val_loss"] = sum(losses) / total_samples

        result["num_samples"] = total_samples

        return result

    def predict(self, pred_iterator):
        # switch to evaluate mode
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

    def forward_batch(self, batch, batch_info):
        """Calculates the loss and accuracy over a given batch.

        You can override this method to provide arbitrary metrics.

        Same as ``train_batch``, this method implementation assumes that
        batches are in (\*features, labels) format by default. So we also
        support multiple inputs model.

        Args:
            batch: One item of the validation iterator.
            batch_info (dict): Contains information per batch from
                ``validate()``.

        Returns:
            A dict of metrics.
                By default, returns "val_loss", "val_accuracy", and
                "num_samples". When overriding, consider returning
                "num_samples" in the metrics because
                by default, ``validate`` uses "num_samples" to
                calculate averages.
        """
        # unpack features into list to support multiple inputs model
        features, target = batch

        # compute output
        with self.timers.record("eval_fwd"):
            if torch.is_tensor(features):
                output = self.model(features)
            elif isinstance(features, (tuple, list)):
                output = self.model(*features)
            else:
                invalidInputError(False,
                                  "Features should be tensor, list/tuple or dict, "
                                  "but got {}".format(type(features)))

            if isinstance(output, tuple) or isinstance(output, list):
                # Then target is also assumed to be a tuple or list.
                loss = self.criterion(*output, *target)
            else:
                loss = self.criterion(output, target)

        return output, target, loss

    def state_dict(self):
        """Override this to return a representation of the operator state.

        Returns:
            dict: The state dict of the operator."""
        pass

    def load_state_dict(self, state_dict):
        """Override this to load the representation of the operator state.

        Args:
            state_dict (dict): State dict as returned by the operator. """
        pass

    @property
    def config(self):
        """dict: Provided into TorchTrainer."""
        return self._config

    @property
    def model(self):
        """First or only model created by the provided ``model_creator``."""
        return self._models[0]

    @property
    def models(self):
        """List of models created by the provided ``model_creator``."""
        return self._models

    @property
    def optimizer(self):
        """First or only optimizer(s) created by the ``optimizer_creator``."""
        return self._optimizers[0]

    @property
    def optimizers(self):
        """List of optimizers created by the ``optimizer_creator``."""
        return self._optimizers

    @property
    def world_rank(self):
        """int: The rank of the parent runner. Always 0 if not distributed."""
        return self._world_rank

    @property
    def criterion(self):
        """Criterion created by the provided ``loss_creator``."""
        return self._criterion

    @property
    def scheduler(self):
        """First or only scheduler(s) created by the ``scheduler_creator``."""
        if self._schedulers:
            return self._schedulers[0]

    @property
    def schedulers(self):
        """List of schedulers created by the ``scheduler_creator``."""
        return self._schedulers

    @property
    def use_fp16(self):
        """bool: Whether the model and optimizer have been FP16 enabled."""
        return self._use_fp16

    @property
    def use_tqdm(self):
        """bool: Whether tqdm progress bars are enabled."""
        return self._use_tqdm
