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
import logging
import shutil
import tensorflow as tf

from bigdl.orca.learn.utils import get_replaced_path
from bigdl.orca.data.file import put_local_files_with_prefix_to_remote, \
    put_local_dir_tree_to_remote, put_local_file_to_remote

logger = logging.getLogger(__name__)


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, tf_callback, rank=None):
        super(Callback, self).__init__()
        self.tf_callback = tf_callback
        self.rank = rank

    def set_params(self, params):
        self.tf_callback.set_params(params)

    def set_model(self, model):
        self.tf_callback.set_model(model)

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""
        self.tf_callback.on_batch_begin()

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
        self.tf_callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`. For training epoch, the values of the
             `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
               0.7}`.
        """
        self.tf_callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        # For backwards compatibility.
        self.tf_callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.tf_callback.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        self.tf_callback.on_test_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.
        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        self.tf_callback.on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently the output of the last call to `on_epoch_end()`
              is passed to this argument for this method but that may change in
              the future.
        """
        self.tf_callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        self.tf_callback.on_predict_end(logs)


class ModelCheckpoint(Callback):
    def __init__(self,
                 tf_callback,
                 rank=None):
        super(ModelCheckpoint, self).__init__(tf_callback, rank)
        self.original_checkpoint_path = self.tf_callback.filepath
        self.original_checkpoint_dir = os.path.dirname(self.original_checkpoint_path)
        self.local_checkpoint_path = get_replaced_path(self.original_checkpoint_path)
        self.tf_callback.filepath = self.local_checkpoint_path
        self.rank = rank
        self.copy_return = 0

    def on_train_batch_end(self, batch, logs=None):
        self.tf_callback.on_train_batch_end(batch, logs)
        if self.tf_callback._should_save_on_batch(batch):
            self.copy_return = self._save_remote_checkpoint()

    def on_epoch_end(self, epoch, logs=None):
        self.tf_callback.on_epoch_end(epoch, logs)
        if self.tf_callback.save_freq == 'epoch':
            self.copy_return = self._save_remote_checkpoint()

    def on_train_end(self, logs=None):
        self.tf_callback.on_train_end(logs)
        if self.copy_return == 0:
            if os.path.exists(os.path.dirname(self.local_checkpoint_path)):
                shutil.rmtree(os.path.dirname(self.local_checkpoint_path))
        else:
            logger.warning("Error when copy local checkpoint {} to {}, "
                           "please get the local checkpoint manually"
                           .format(self.local_checkpoint_path, self.original_checkpoint_path))

    def _save_remote_checkpoint(self):
        if self.rank is not None:
            if self.rank == 0:
                write_filepath = self.tf_callback._write_filepath
                if write_filepath.endswith(".h5") or write_filepath.endswith(".keras"):
                    return put_local_file_to_remote(write_filepath,
                                                    os.path.join(self.original_checkpoint_dir,
                                                                 os.path.basename(write_filepath))
                                                    )
                else:
                    if self.tf_callback.save_weights_only:
                        # copy checkpoint data files
                        ret = put_local_files_with_prefix_to_remote(
                            write_filepath, self.original_checkpoint_dir)
                        if ret == 0:
                            # copy "checkpoint" file
                            return put_local_file_to_remote(
                                os.path.join(os.path.dirname(write_filepath), "checkpoint"),
                                os.path.join(self.original_checkpoint_dir, "checkpoint")
                            )
                        return ret

                    else:
                        return put_local_dir_tree_to_remote(
                            write_filepath,
                            os.path.join(self.original_checkpoint_dir,
                                         os.path.basename(write_filepath)
                                         )
                        )
        return 0
