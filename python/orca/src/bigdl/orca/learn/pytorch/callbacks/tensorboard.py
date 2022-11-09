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
import os
import shutil
import tempfile

from torch.utils.tensorboard import SummaryWriter

from bigdl.orca.data.file import put_local_dir_tree_to_remote
from bigdl.orca.learn.pytorch.callbacks import Callback
from bigdl.dllib.utils.log4Error import invalidInputError


class TensorBoardCallback(Callback):

    def __init__(
        self,
        log_dir=None,
        freq="epoch",
        **kwargs,
    ):
        """
        :param log_dir: Log directory of TensorBoard.
        :param freq: Frequency of logging metrics and loss.
            Accept values: 'batch' or 'epoch' or integer. When using 'batch',
            writes the losses and metrics to TensorBoard after each batch.
            The same applies for 'epoch'. If using an integer, let's say 1000,
            the callback will write the metrics and losses to TensorBoard every 1000 batches.
            Note that writing too frequently to TensorBoard can slow down your training.
        :param **kwargs: The keyword arguments will be pased to ``SummaryWriter``.
        """
        self.log_dir = log_dir
        self.tmp_dir = os.path.join(tempfile.mkdtemp(), os.path.basename(log_dir))
        self.freq = freq
        self.kwargs = kwargs
        self.unlog_items = ["epoch", "batch_count", "num_samples"]
        super().__init__()

    def on_batch_begin(self, batch):
        """
        Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        :param batch: Integer, index of batch within the current epoch.
        """
        pass

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        :param batch: Integer, index of batch within the current epoch.
        :param logs: Dict. Aggregated metric results up until this batch.
        """
        if self.freq != "epoch" and self._is_rank_zero():
            if self.freq == "batch" or batch % int(self.freq) == 0:
                writer = SummaryWriter(log_dir=self.tmp_dir, **self.kwargs)
                for name, value in logs.items():
                    if name not in self.unlog_items:
                        writer.add_scalar(name, value, batch)
                writer.close()

    def on_epoch_begin(self, epoch):
        """
        Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently, saved stats in last epoch has been passed to this argument
        for this method but may change in the future.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        :param epoch:  Integer, index of epoch.
        :param logs: Dict, metric results for this training epoch, and for the validation epoch if
            validation is performed. Validation result keys are prefixed with val_. For training
            epoch, the values of the Model's metrics are returned.
            Example : {'loss': 0.2, 'accuracy': 0.7}
        """
        if self.freq == "epoch" and self._is_rank_zero():
            writer = SummaryWriter(log_dir=self.tmp_dir, **self.kwargs)
            for name, value in logs.items():
                if name not in self.unlog_items:
                    writer.add_scalar(name, value, epoch)
            writer.close()

    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        :param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        """
        if self._is_rank_zero():
            put_local_dir_tree_to_remote(self.tmp_dir, self.log_dir)
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)

    def set_model(self, model):
        self.model = model

    def set_param(self, param):
        self.params = param

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _is_rank_zero(self):
        invalidInputError(self.trainer, "Sanity check failed. Must call set_trainer first!")
        rank = self.trainer.rank
        return rank == 0
