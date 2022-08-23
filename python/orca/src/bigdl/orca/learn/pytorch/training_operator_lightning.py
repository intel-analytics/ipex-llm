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

from bigdl.orca.learn.pytorch.utils import (TimerCollection, AverageMeterCollection,
                                            NUM_SAMPLES)
from bigdl.orca.learn.pytorch.constants import (SCHEDULER_STEP_EPOCH, NUM_STEPS,
                                                SCHEDULER_STEP_BATCH, SCHEDULER_STEP)
from bigdl.dllib.utils.log4Error import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback as callback_lightning
from bigdl.orca.learn.pytorch.training_operator import TrainingOperator

tqdm = None
try:
    from tqdm import tqdm
except ImportError:
    pass

def _is_multiple(component):
    """Checks if a component (optimizer, model, etc) is not singular."""
    return isinstance(component, collections.Iterable) and len(component) > 1


class TrainingOperatorLightning(TrainingOperator):
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
                 models_ori,
                 models,
                 world_rank,
                 use_tqdm=False,
                 sync_stats=False):
        TrainingOperator.__init__(self, config, models, [], world_rank, None, None, False, use_tqdm, sync_stats, "torch")
        # You are not expected to override this method.
        self._models = models  # List of models
        self._models_ori = models_ori  # List of models
        invalidInputError(isinstance(models, collections.Iterable),
                          "Components need to be iterable. Got: {}".format(type(models)))
        self._world_rank = world_rank

        self._config = config
        if tqdm is None and use_tqdm:
            invalidInputError(False,
                              "tqdm must be installed to use tqdm in training.")
        self._use_tqdm = use_tqdm
        self.global_step = 0
        self.sync_stats = sync_stats
        self.dist_backend = "torch"
        self.timers = TimerCollection()
        self.optimizer_config = self.model_ori.configure_optimizers()
        self.setup(config)


    def _train_loop(self, iterator, info, _progress_bar, metric_meters, callbacks):
        for batch_idx, batch in enumerate(iterator):
            batch_info = {
                "batch_idx": batch_idx,
                "global_step": self.global_step
            }
            batch_info.update(info)
            if callbacks is not None:
                for callback in callbacks:
                    if isinstance(callback, callback_lightning):
                        callback.on_train_batch_start(pl.Trainer(), self.model_ori, batch, batch_info["batch_idx"])
                    else:
                        callback.on_batch_begin(batch_idx)
            metrics = self.train_batch(batch, batch_info=batch_info)
            if self.use_tqdm and self.world_rank == 0:
                _progress_bar.n = batch_idx + 1
                postfix = {}
                if "train_loss" in metrics:
                    postfix.update(loss=metrics["train_loss"])
                _progress_bar.set_postfix(postfix)

            if self.scheduler and batch_info.get(
                    SCHEDULER_STEP) == SCHEDULER_STEP_BATCH:
                self.scheduler.step()

            metric_meters.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
            self.global_step += 1
            if callbacks is not None:
                for callback in callbacks:
                    if isinstance(callback, callback_lightning):
                        callback.on_train_batch_end(pl.Trainer(), self.model_ori, metrics["train_loss"], batch, batch_info["batch_idx"])
                    else:
                        callback.on_batch_end(batch_idx)

    def train_batch(self, batch, batch_info):
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
            batch_info (dict): Information dict passed in from ``train_epoch``.
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
        *features, target = batch

        # Compute output.
        with self.timers.record("fwd"):
            training_step_output = self.model_ori.training_step(batch, batch_info['batch_idx'])
            if isinstance(training_step_output, dict):
                invalidInputError("loss" in training_step_output, "training_step function must has 'loss' output")
                loss = training_step_output['loss']
            opt = self.optimizer[0]
            ind = 0
            opt_len = len(self.optimizer)
            if opt_len > 1:
                freq = 0
                freq_interval = []
                for i in range(opt_len):
                    freq += self.optimizer[i]["frequency"]
                    freq_interval.append(freq)
                for i in range(len(freq_interval)):
                    if freq_interval[i] > batch_info['batch_idx'] % freq:
                        ind = i
                        break
                opt = self.optimizer[ind]["optimizer"]

        # Compute gradients in a backward pass.
        with self.timers.record("grad"):
            opt.zero_grad()
            loss.backward()

        # Call step of optimizer to update model params.
        with self.timers.record("apply"):
            opt.step()
        return {"train_loss": loss.item(), NUM_SAMPLES: target.size(0)}


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
        *features, target = batch

        # # compute output
        with self.timers.record("eval_fwd"):
            validation_step_output = self.model_ori.validation_step(batch, batch_info['batch_idx'])
            if isinstance(validation_step_output, dict):
                invalidInputError("loss" in validation_step_output, "training_step function must has 'loss' output")
                loss = validation_step_output['loss']
                output = validation_step_output['predictions']
            else:
                output = self.model(*features)

        return output, target, loss


    @property
    def model_ori(self):
        """First or only model created by the provided ``model_creator``."""
        return self._models_ori[0]

    @property
    def models_ori(self):
        """List of models created by the provided ``model_creator``."""
        return self._models_ori

    @property
    def optimizer(self):
        if isinstance(self.optimizer_config, tuple):
            return self.optimizer_config[0]
        elif not isinstance(self.optimizer_config, collections.Iterable):
            return [self.optimizer_config]
        else:
            return self.optimizer_config

    @property
    def scheduler(self):
        if not isinstance(self.optimizer_config, tuple):
            return None
        return self.optimizer_config[1][0]
