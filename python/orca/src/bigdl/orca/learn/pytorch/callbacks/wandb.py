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
from bigdl.orca.learn.pytorch.callbacks import Callback
from bigdl.dllib.utils.log4Error import invalidInputError
import copy

try:
    import wandb
except ImportError:
    invalidInputError(False, "pip install 'wandb' to use WandbLoggerCallback")
    wandb = None


class WandbLoggerCallback(Callback):

    def __init__(
        self,
        project: str,
        log_config: bool = False,
        watch_model: bool = False,
        **kwargs,
    ):
        """
        Weights and biases (https://www.wandb.ai/) is a tool for experiment tracking,
        model optimization, and dataset versioning.

        WandbLoggerCallback automatically uses Weights and biases to log metric results at the end
        of each epoch. User could choose whether to watch model gradients and whether to log
        hyper-parameters defined in config.

        Example:
            >>> wnb_callback = WandbLoggerCallback(
            ...     project='my_project',
            ... )
        :param project: Name of the Wandb project. Mandatory.
        :param log_config: Whether to log ``config``.
        :param watch_model: Whether to log model gradients and model topology.
        :param **kwargs: The keyword arguments will be pased to ``wandb.init()``.
        """
        try:
            wandb.ensure_configured()
        except AttributeError:
            invalidInputError(False, "You should run `wandb login` from the command line first.")
        self.project = project
        self.log_config = log_config
        self.watch_model = watch_model
        self.kwargs = kwargs
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch:  Integer, index of epoch.
        @param logs: Dict, metric results for this training epoch, and for the validation epoch if
            validation is performed. Validation result keys are prefixed with val_. For training
            epoch, the values of the Model's metrics are returned.
            Example : {'loss': 0.2, 'accuracy': 0.7}
        """
        self.run.log(logs)

    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        @param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        is_rank_zero = self._is_rank_zero()
        if "config" in self.kwargs:
            config = self.kwargs.pop("configs")
        else:
            config = {}
        if self.log_config:
            trainer_config = copy.copy(self.trainer.config)
            config.update(trainer_config)

        if is_rank_zero:
            self.run = wandb.init(
                project=self.project,
                config=config,
                **self.kwargs
            )
            if self.watch_model:
                self.run.watch(self.model)
        else:
            self.run = None

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        :param logs: Dict. Currently the output of the last call to on_epoch_end() is passed to
            this argument for this method but that may change in the future.
        """
        self.run.finish()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _is_rank_zero(self):
        invalidInputError(self.trainer, "Sanity check failed. Must call set_trainer first!")
        rank = self.trainer.rank
        return rank == 0
