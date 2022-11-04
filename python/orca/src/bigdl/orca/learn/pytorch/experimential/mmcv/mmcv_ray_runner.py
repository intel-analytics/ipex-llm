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

from mmcv.runner import EpochBasedRunner
from mmcv.parallel.distributed import MMDistributedDataParallel
from bigdl.orca.learn.pytorch.experimential.core.base_ray_runner import BaseRayRunner

from typing import (Any, Dict, List, Optional, Tuple, Callable, overload)


class MMCVRayRunner(BaseRayRunner, EpochBasedRunner):
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
        self.model = MMDistributedDataParallel(self.model)

    def train_epochs(self,
                     data_loaders_creators: List[Callable],
                     workflow: List[Tuple[str, int]],
                     max_epochs: Optional[int] = None,  # deprecated
                     **kwargs):
        data_loaders = [self.with_sampler(creator(self.config)) for
                        creator in data_loaders_creators]
        super().run(data_loaders, workflow, max_epochs, **kwargs)

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
