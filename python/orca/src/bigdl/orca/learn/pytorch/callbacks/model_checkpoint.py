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
import os
import re
import warnings

from bigdl.orca.learn.pytorch.utils import get_filesystem
from bigdl.dllib.utils.log4Error import invalidInputError


class ModelCheckpoint(Callback):

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"

    def __init__(self,
                 filepath=None,
                 save_weights_only=False,
                 ):
        """
        ModelCheckpoint callback is used in conjunction with training using estimator.fit() to save
        a model or weights (in a checkpoint file) at some interval, so the model or weights can be
        loaded later to continue the training from the state saved.
        Example:
            >>> checkpoint_callback = ModelCheckpoint(
            ...     filepath='my/path/sample-mnist-{epoch:02d}',
            ... )
        And checkpoints will be saved as file with path like 'my/path/sample-mnist-epoch=1.ckpt'
        with different epoch values.
        :param filepath: path to save the model file.
        """
        super().__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.last_ckpt_path = ""
        self.filename = os.path.basename(self.filepath)
        self.dirname = os.path.dirname(self.filepath)

    def on_batch_begin(self, batch):
        """
        Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    def on_batch_end(self, batch):
        """
        Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    def on_epoch_begin(self, epoch):
        """
        Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch: Integer, index of epoch.
        @param logs: Dict. Currently, saved stats in last epoch has been passed to this argument
        for this method but may change in the future.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch:  Integer, index of epoch.
        """
        stats = {"epoch": self.trainer.epochs}
        last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname,
                                                      filename=self.filename,
                                                      stats=stats)
        self.trainer.save_checkpoint(last_ckpt_path, self.save_weights_only)

    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        @param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        # todo: support resume training
        dirname = os.path.dirname(self.filepath)
        fs = get_filesystem(dirname)
        if fs.exists(dirname):
            files = [os.path.basename(f["name"]) for f in fs.listdir(dirname)]
            files = [x for x in files if "ckpt" in x]
            if len(files) == 0:
                return None
            invalidInputError(False,
                              f"Find non-empty dirname with filepath of {self.filepath}.")
        else:
            fs.mkdirs(dirname)

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        """
        stats = {"epoch": self.trainer.epochs}
        last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname,
                                                      filename=self.CHECKPOINT_NAME_LAST,
                                                      stats=stats)
        previous, self.last_ckpt_path = self.last_ckpt_path, last_ckpt_path
        self.trainer.save_checkpoint(last_ckpt_path, self.save_weights_only)
        if previous and previous != last_ckpt_path:
            self.trainer.remove_checkpoint(previous)

    def set_model(self, model):
        self.model = model

    def set_param(self, param):
        self.params = param

    def set_trainer(self, trainer):
        self.trainer = trainer

    @classmethod
    def get_latest_checkpoint(cls, dirname):
        """
        Finds the filepath of latest saved checkpoint file.
        :param dirname: directory where the checkpoints were saved
        return: The full path to the latest checkpoint or `None` if no checkpoint was found.
        """
        ckpt_path = cls._format_checkpoint_name(dirname, filename=cls.CHECKPOINT_NAME_LAST)
        fs = get_filesystem(ckpt_path)
        if not fs.exists(ckpt_path):
            invalidInputError(False,
                              f"Latest checkpoint at {ckpt_path} not found.")
        return ckpt_path

    @classmethod
    def _format_checkpoint_name(cls, dirname, filename, stats=None):
        """
        checkpoint name is in the format of 'epoch={epoch}.ckpt'
        """
        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            stats = dict() if stats is None else stats
            for group in groups:
                name = group[1:]

                if "epoch" not in name:
                    warnings.warn("We only support filepath with {epoch} for now.")

                filename = filename.replace(group, name + "={" + name)

                if name not in stats:
                    stats[name] = 0
            filename = filename.format(**stats)
        ckpt_name = f"{filename}{cls.FILE_EXTENSION}"
        ckpt_path = os.path.join(dirname, ckpt_name)
        return ckpt_path
