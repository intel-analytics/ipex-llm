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
import re
import tempfile
import time
import mmcv
import numpy as np
import torch
import warnings
from collections import OrderedDict
from torch.optim import Optimizer
from mmcv.runner import EpochBasedRunner, DistEvalHook
from mmcv.runner.utils import get_host_info
from mmcv.parallel import is_module_wrapper
from mmcv.fileio.file_client import BaseStorageBackend
from mmcv.utils.misc import is_list_of
from mmcv.runner.checkpoint import CheckpointLoader
from bigdl.orca.learn.pytorch.utils import get_batchsize
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.learn.pytorch.core.base_runner import BaseRunner
from .distributed import MMDistributedDataParallel

from typing import (TYPE_CHECKING, Union, Any, Dict, List, Optional, Tuple, Callable)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class HDFSBackend(BaseStorageBackend):
    """
    HDFS storage backend for saving ckpt

    This backend will be used when runner contains a CheckpointHook and
    CheckpointHook's out_dir starts with hdfs://
    """

    def get(self, filepath: str) -> bytes:
        temp_file = self._hdfs_to_local(filepath)

        with open(temp_file, 'rb') as f:
            value_buf = f.read()
        os.remove(temp_file)
        return value_buf

    def get_text(self, filepath: str, encoding: str = 'utf-8') -> str:
        temp_file = self._hdfs_to_local(filepath)

        with open(temp_file, encoding=encoding) as f:
            value_buf = f.read()
        os.remove(temp_file)
        return value_buf

    def put(self, obj: bytes, filepath: str) -> None:
        filename = os.path.basename(filepath)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(obj)

        self._local_to_hdfs(temp_path, filepath)
        os.remove(temp_path)

    def put_text(self, obj: str, filepath: str, encoding: str = 'utf-8') -> None:
        filename = os.path.basename(filepath)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'w', encoding=encoding) as f:
            f.write(obj)

        self._local_to_hdfs(temp_path, filepath)
        os.remove(temp_path)

    def join_path(self, filepath: str,
                  *filepaths: str) -> str:
        return os.path.join(filepath, *filepaths)

    def _hdfs_to_local(self, filepath: str) -> str:
        import uuid
        from bigdl.dllib.utils.file_utils import append_suffix
        from bigdl.orca.data.file import get_remote_file_to_local
        file_name = str(uuid.uuid1())
        file_name = append_suffix(file_name, filepath)
        temp_path = os.path.join(tempfile.gettempdir(), file_name)

        get_remote_file_to_local(filepath, temp_path)
        return temp_path

    def _local_to_hdfs(self, local_path: str, hdfs_path: str) -> None:
        from bigdl.orca.data.file import exists, makedirs, put_local_file_to_remote
        work_dir = os.path.dirname(hdfs_path)
        if not exists(work_dir):
            makedirs(work_dir)

        put_local_file_to_remote(local_path, hdfs_path)


@CheckpointLoader.register_scheme(prefixes='hdfs://')
def load_from_hdfs(filename: str,
                   map_location: Union[str, Callable, None]=None) -> Union[dict, OrderedDict]:
    """
    load checkpoint by HDFS file path

    Args:
        filename (str): HDFS checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    import uuid
    from bigdl.dllib.utils.file_utils import append_suffix
    from bigdl.orca.data.file import exists, get_remote_file_to_local
    if not exists(filename):
        invalidInputError(False, f"checkpoint at {filename} not found.")

    temp_file_name = append_suffix(str(uuid.uuid1()), filename)
    temp_path = os.path.join(tempfile.gettempdir(), temp_file_name)
    get_remote_file_to_local(filename, temp_path)

    checkpoint = torch.load(temp_path)
    return checkpoint


class MMCVRayEpochRunner(BaseRunner, EpochBasedRunner):
    EBR_slots = (
        "model",
        "batch_processor",
        "optimizer",
        "logger",
        "meta",
        "work_dir",
        "_model_name",
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

    def __init__(self,
                 mmcv_runner_creator: Callable,
                 config: Optional[Dict]=None) -> None:
        self.mmcv_runner_creator = mmcv_runner_creator
        self.config = config
        self._backend = "torch-local"

    def setup_components(self) -> None:
        runner = self.mmcv_runner_creator(self.config)
        self._wrap_from_ebr(runner)

    def setup_ddp_components(self) -> None:
        # MMDDP is implemented by MMCV, it has two main differences with PyTorch DDP:
        # 1. It supports a custom type :class:`DataContainer` which allows more
        #    flexible control of input data.
        # 2. It implement two APIs ``train_step()`` and ``val_step()``.
        self.model = MMDistributedDataParallel(self.model)  # type:ignore

    def train_epochs(self,
                     data_loaders_creators: List[Callable],
                     workflow: List[Tuple[str, int]],
                     max_epochs: Optional[int] = None,  # deprecated
                     **kwargs) -> List[Dict]:
        data_loaders = [self.with_sampler(creator(self.config)) for
                        creator in data_loaders_creators]
        # runner was created in method setup_components, which may contains the
        # creation of DistEvalHook. At that time, ddp env was not initialized, so
        # the val_dataloder in DistEvalHook is not correctly built with a sampler.
        # so add sampler for val_dataloder if there is a DistEvalHook.
        for hook in self._hooks:
            if isinstance(hook, DistEvalHook):
                # don't shuffle the val data
                hook.dataloader = self.with_sampler(hook.dataloader, shuffle=False)

        return self.run(data_loaders, workflow, max_epochs, **kwargs)

    def run(self,
            data_loaders: List['DataLoader'],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> List[Dict]:
        invalidInputError(isinstance(data_loaders, list), "data_loaders should be a list")
        invalidInputError(is_list_of(workflow, tuple), "workflow shoud be a list of tuple")
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

    def train(self, data_loader: 'DataLoader', **kwargs) -> Dict:
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

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        EpochBasedRunner.save_checkpoint(self, out_dir, filename_tmpl,
                                         save_optimizer, meta, create_symlink)

    def load_checkpoint(
            self,
            filename: str,
            map_location: Union[str, Callable]='cpu',
            strict: bool = False,
            revise_keys: List = [(r'^module.', '')],
    ) -> Union[Dict, OrderedDict]:
        """Load checkpoint from a file or URI.

        Args:
            filename (str): HDFS file path, ``hdfs://xxx``.
            map_location (str): Same as :func:`torch.load`.
            strict (bool): Whether to allow different params for the model and
                checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Default: strip
                the prefix 'module.' by [(r'^module\\.', '')].

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint = CheckpointLoader.load_checkpoint(filename, map_location, self.logger)
        self.load_state_dict(checkpoint, strict, revise_keys)
        return checkpoint

    def remove_checkpoint(self, filepath: str) -> None:
        pass

    def get_state_dict(self) -> Dict:
        """Returns the state of the runner."""
        meta = {}
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch, iter=self.iter)
        model = self.model
        if is_module_wrapper(model):
            model = model.module

        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            meta.update(CLASSES=model.CLASSES)

        from mmcv.runner.checkpoint import get_state_dict as model_state_dict
        state = {
            'meta': meta,
            'state_dict': model_state_dict(model)
        }

        if isinstance(self.optimizer, Optimizer):
            state['optimizer'] = self.optimizer.state_dict()
        elif isinstance(self.optimizer, dict):
            state['optimizer'] = {}
            for name, optim in self.optimizer.items():
                state['optimizer'][name] = optim.state_dict()

        return state

    def load_state_dict(self, checkpoint: Dict, strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]) -> None:
        """Sets the state of the model."""

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

            # strip prefix of state_dict
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            for p, r in revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k): v
                     for k, v in state_dict.items()})
            # Keep metadata in state_dict
            state_dict._metadata = metadata

        from mmcv.runner.checkpoint import load_state_dict
        load_state_dict(self.model, state_dict, strict, self.logger)

    def shutdown(self) -> None:
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
