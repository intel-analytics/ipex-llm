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

import types
import copy
import ray

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.learn.pytorch.core.base_ray_estimator import BaseRayEstimator
from bigdl.orca.learn.pytorch.experimential.mmcv.mmcv_ray_runner import MMCVRayEpochRunner
from bigdl.orca.learn.pytorch.utils import process_stats

from typing import (Dict, List, Optional, Tuple, Callable, Union)


class MMCVRayEstimator(BaseRayEstimator):
    def __init__(self,
                 *,
                 mmcv_runner_creator: Callable,
                 backend: str = "ray",
                 workers_per_node: int = 1,
                 config: Optional[Dict]=None) -> None:
        if not (isinstance(mmcv_runner_creator, types.FunctionType)):
            invalidInputError(False, "Must provide a function for mmcv_runner_creator")

        self.mmcv_runner_creator = mmcv_runner_creator
        self.backend = backend
        self.runner_cls = MMCVRayEpochRunner
        self.config = {} if config is None else config
        worker_config = copy.copy(self.config)
        params = dict(
            mmcv_runner_creator=self.mmcv_runner_creator,
            config=worker_config
        )
        self.setup(params, self.backend, self.runner_cls, workers_per_node)

    def fit(self,
            data_loaders_creators: List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,  # deprecated
            reduce_results: bool = True,
            **kwargs) -> List:
        """Trains a MMCV model given training and val data for several epochs.

        :param data_loaders_creators: Dataloader creators for training and validation.
        :param workflow: A list of (phase, epochs) to specify the
               running order and epochs. E.g, [('train', 2), ('val', 1)] means
               running 2 epochs for training and 1 epoch for validation,
               iteratively.
        :param max_epochs: Set max_epochs for MMCV runner is deprecated
        :param reduce_results: Whether to average all metrics across all workers into
               one dict. If a metric is a non-numerical value, the one value will be randomly
               selected among the workers. If False, returns a list of dicts for
               all workers. Default is True.
        """
        for creator in data_loaders_creators:
            if not (isinstance(creator, types.FunctionType)):
                invalidInputError(False, "Must provide a function for all dataloader creator")

        params = dict(data_loaders_creators=data_loaders_creators,
                      workflow=workflow,
                      max_epochs=max_epochs,
                      **kwargs)

        self.setup_torch_ddp()

        success, worker_stats = self._train_epochs(**params)

        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    def run(self,
            data_loaders_creators: List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,  # deprecated
            reduce_results: bool = True,
            **kwargs) -> List:
        """
        Same as fit method, the parameters are consistent with MMCV runner.run()
        """
        return self.fit(data_loaders_creators, workflow, max_epochs, reduce_results, **kwargs)

    def predict(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def get_model(self) -> Dict:
        state = self.get_state_dict()
        model_state = state["state_dict"]
        return model_state

    def load_checkpoint(
            self,
            filename: str,
            map_location: Union[str, Callable]='cpu',
            strict: bool = False,
            revise_keys: List = [(r'^module.', '')],
    ) -> None:
        """Load checkpoint from a file or URI. The filename should either be a
        local path on driver or a HDFS path

        Args:
            filename (str): Local path on Driver or HDFS path.
            map_location (str): Same as :func:`torch.load`.
            strict (bool): Whether to allow different params for the model and
                checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Default: strip
                the prefix 'module.' by [(r'^module\\.', '')].

        Returns:
            None
        """
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(filename):
            self.load(filename)
        else:
            params = dict(
                filename=filename,
                map_location=map_location,
                strict=strict,
                revise_keys=revise_keys
            )
            results = [
                worker.load_checkpoint.remote(**params)
                for worker in self.remote_workers
            ]
            ray.get(results)
