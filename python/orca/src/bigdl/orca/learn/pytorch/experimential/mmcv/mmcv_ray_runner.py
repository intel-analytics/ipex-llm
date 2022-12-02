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

import time
import mmcv
import numpy as np
import torch
import warnings
from mmcv.runner import EpochBasedRunner
from mmcv.runner.utils import get_host_info
from mmcv.parallel.distributed import MMDistributedDataParallel
from bigdl.orca.learn.pytorch.utils import get_batchsize
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.learn.pytorch.experimential.core.base_ray_runner import BaseRayRunner

from typing import (Any, Dict, List, Optional, Tuple, Callable, overload)


class MMCVRayEpochRunner(BaseRayRunner, EpochBasedRunner):
    EBR_slots = (
        "model",
        "batch_processor",
        "optimizer",
        "logger",
        "meta",
        "work_dir",
        "_model_name",
        "_rank",
        "_world_size",
        "timestamp",
        "mode",
        "_hooks",
        "_epoch",
        "_iter",
        "_inner_iter",
        "_max_epochs",
        "_max_iters",
        "log_buffer",
    )

    def __init__(self, mmcv_runner_creator=None, config=None):
        self.mmcv_runner_creator = mmcv_runner_creator
        self.config = config
        self._backend = "torch-local"

    def setup_components(self):
        runner = self.mmcv_runner_creator(self.config)
        self._wrap_from_ebr(runner)
        # MMDDP is implemented by MMCV, it has two main differences with PyTorch DDP:
        # 1. It supports a custom type :class:`DataContainer` which allows more
        #    flexible control of input data.
        # 2. It implement two APIs ``train_step()`` and ``val_step()``.
        self.model = MMDistributedDataParallel(self.model)

    def train_epochs(self,
                     data_loaders_creators: List[Callable],
                     workflow: List[Tuple[str, int]],
                     max_epochs: Optional[int] = None,  # deprecated
                     **kwargs):
        data_loaders = [self.with_sampler(creator(self.config)) for
                        creator in data_loaders_creators]
        return self.run(data_loaders, workflow, max_epochs, **kwargs)

    def run(self,
            data_loaders: List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> List:
        invalidInputError(isinstance(data_loaders, list), "data_loaders should be a list")
        invalidInputError(mmcv.is_list_of(workflow, tuple), "workflow shoud be a list of tuple")
        invalidInputError(len(data_loaders) == len(workflow),
                          "data_loaders and workflow should have the same length")

        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        invalidInputError(self._max_epochs is not None,
                          "max_epochs must be specified during instantiation")

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        stats_list = list()

        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        invalidInputError(False,
                                          f'runner has no method named "{mode}" to run an epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    invalidInputError(False, 'mode in workflow must be a str, '
                                             'but got {}'.format(type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    train_stats = epoch_runner(data_loaders[i], **kwargs)
                    stats = dict(epoch=self.epoch, **train_stats)
                    stats_list.append(stats)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        return stats_list

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        if self.log_buffer.ready:
            self.logger.warning("log_buffer is not cleared, this may cause the return "
                                "value of fit/run method is not correct")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        # do not call log_buffer.clear() during an epoch,
        # otherwise, the dirver will not be able to fetch the data.
        stats = self._get_epoch_stats()
        self.call_hook('after_train_epoch')
        self._epoch += 1
        return stats

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            invalidInputError(False,
                              '"batch_processor()" or "model.train_step()" '
                              'and "model.val_step()" must return a dict')

        if 'log_vars' in outputs:
            # ensure the key 'loss' is in the log_vars, so the driver can fetch
            # loss infos from log_buffer, if log_vars['loss'] is not existed, add
            # it to log_vars from outputs['loss']
            if 'loss' in outputs['log_vars']:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            else:
                log_vars = outputs['log_vars'].copy()
                log_vars['loss'] = outputs['loss'].item()
                self.log_buffer.update(log_vars, outputs['num_samples'])
        else:
            log_vars = dict()
            log_vars['loss'] = outputs['loss'].item()
            self.log_buffer.update(log_vars, get_batchsize(data_batch))

        self.outputs = outputs

    def predict(self, **kwargs):
        pass

    def validate(self, **kwargs):
        pass

    def get_state_dict(self):
        """Returns the state of the runner."""
        pass

    def load_state_dict(self, state):
        """Sets the state of the model."""
        pass

    def _save_checkpoint(self, filepath, save_weights_only=False):
        """Save checkpoint."""
        pass

    def shutdown(self):
        pass

    def _wrap_from_ebr(self, epoch_based_runner):
        for attr in self.EBR_slots:
            # todo: check necessary components
            setattr(self, attr, getattr(epoch_based_runner, attr))

    def _get_epoch_stats(self):
        # calculate epoch average stats
        self.log_buffer.average()
        batch_count = len(self.log_buffer.val_history['loss'])
        num_samples = np.sum(self.log_buffer.n_history['loss'])
        last_val_dict = dict()
        for key, vals in self.log_buffer.val_history.items():
            last_val_dict["last_" + str(key)] = vals[-1]

        stats = dict(batch_count=batch_count, num_samples=num_samples,
                     **self.log_buffer.output, **last_val_dict)
        return stats

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank):
        self._rank = rank

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend

    @property
    def size(self):
        return self._world_size

    @size.setter
    def size(self, size):
        self._world_size = size
