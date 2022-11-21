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


from bigdl.orca.learn.ray_estimator import Estimator
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.learn.pytorch.pytorch_ray_worker import PytorchRayWorker
from bigdl.orca.data.file import enable_multi_fs_save, enable_multi_fs_load

import numbers
import numpy as np
import io
import torch

import ray
from ray.exceptions import RayActorError
from abc import abstractmethod, ABCMeta

from bigdl.orca.learn.pytorch.utils import find_free_port
from bigdl.dllib.utils.log4Error import invalidInputError

# TODO: full path when merge
from ..utils import check_for_failure, get_driver_node_ip


class BaseRayEstimator(Estimator, metaclass=ABCMeta):
    def __init__(self, backend='ray', runner_cls=PytorchRayWorker, workers_per_node=1, **kwargs):
        # Implement logic to struct parameters
        #
        # self.remote_workers=[]
        # self.setup(params, backend, runner_cls, workers_per_node, workers_per_node)
        pass

    @abstractmethod
    def fit(self, **kwargs):
        """
        Train the model with train data.

        :return: predicted result.
        """

        # Need to preprocess params as pytorch_ray_estimator does.
        #
        # ...
        # self._train_epochs(**kwargs)
        pass

    @abstractmethod
    def predict(self, **kwargs):
        """
        Predict input data.

        :return: predicted result.
        """
        # Need to preprocess params as pytorch_ray_estimator does.
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        """
        Evaluate model.

        :return: evaluation result as a dictionary of {'metric name': metric value}
        """
        # Need to preprocess params as pytorch_ray_estimator does.
        pass

    @abstractmethod
    def get_model(self):
        """
        Get the trained model.

        :return: Trained model
        """
        # Need to preprocess params as pytorch_ray_estimator does.
        pass

    def setup(self, params, backend='ray', runner_cls=PytorchRayWorker, workers_per_node=1):
        ray_ctx = OrcaRayContext.get()
        if backend == "ray":
            import torch.distributed as dist
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=cores_per_node)(runner_cls)
            self.remote_workers = [
                RemoteRunner.remote(**params) for i in range(num_nodes)
            ]
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            driver_ip = get_driver_node_ip()
            driver_tcp_store_port = find_free_port()

            _ = dist.TCPStore(driver_ip, driver_tcp_store_port, -1, True,
                              dist.constants.default_pg_timeout)

            ray.get([
                worker.setup_torch_distribute.remote(
                    driver_ip, driver_tcp_store_port, i, num_nodes)
                for i, worker in enumerate(self.remote_workers)
            ])

        elif backend == "horovod":
            from bigdl.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            self.horovod_runner = HorovodRayRunner(ray_ctx,
                                                   worker_cls=runner_cls,
                                                   worker_param=params,
                                                   workers_per_node=workers_per_node)
            self.remote_workers = self.horovod_runner.remote_workers
            cores_per_node = self.horovod_runner.cores_per_node
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)
            ])
        else:
            invalidInputError(False,
                              "Only \"ray\" and \"horovod\" are supported "
                              "values of backend, but got {}".format(backend))
        self.num_workers = len(self.remote_workers)

    def get_state_dict(self):
        stream_ids = [
            worker.get_state_stream.remote()
            for worker in self.remote_workers
        ]
        # get the first task id that finished executing.
        [stream_id], stream_ids = ray.wait(stream_ids, num_returns=1, timeout=None)
        byte_obj = ray.get(stream_id)
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(
            _buffer,
            map_location="cpu")
        return state_dict

    def load_state_dict(self, state_dict, blocking=True):
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        state_stream = _buffer.getvalue()
        state_id = ray.put(state_stream)

        remote_calls = [
            worker.load_state_stream.remote(state_id)
            for worker in self.remote_workers
        ]
        if blocking:
            ray.get(remote_calls)

    @enable_multi_fs_save
    def save(self, model_path):
        """
        Saves the Estimator state (including model and optimizer) to the provided model_path.

        :param model_path: (str) Path to save the model.
        :return:
        """
        state_dict = self.get_state_dict()
        torch.save(state_dict, model_path)
        return model_path

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model.
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def save_checkpoint(self, model_path):
        """

        Manually saves the Estimator state (including model and optimizer) to the provided
        model_path.

        :param model_path: (str) Path to save the model. Both local and remote path are supported.
               e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.save(model_path)
        else:
            results = [
                worker.save_checkpoint.remote(model_path)
                for worker in self.remote_workers
            ]
            ray.get(results)

    def load_checkpoint(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model. Both local and remote path are
               supported. e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.load(model_path)
        else:
            results = [
                worker.load_checkpoint.remote(model_path)
                for worker in self.remote_workers
            ]
            ray.get(results)

    def shutdown(self, force=False):
        """
        Shuts down workers and releases resources.

        :return:
        """
        if not force:
            cleanup = [
                worker.shutdown.remote() for worker in self.remote_workers
            ]
            try:
                ray.get(cleanup)
                [
                    worker.__ray_terminate__.remote()
                    for worker in self.remote_workers
                ]
            except RayActorError:
                logger.warning(
                    "Failed to shutdown gracefully, forcing a shutdown.")

                for worker in self.remote_workers:
                    logger.warning("Killing worker {}.".format(worker))
                    ray.kill(worker)
        else:
            for worker in self.remote_workers:
                logger.debug("Killing worker {}.".format(worker))
                ray.kill(worker)

        self.remote_workers = []

    def _process_stats(self, worker_stats):
        stats = {
            "num_samples": sum(
                stats.pop("num_samples", np.nan) for stats in worker_stats)
        }

        for stat_key in worker_stats[0]:
            if isinstance(worker_stats[0], numbers.Number):
                stats[stat_key] = np.nanmean(
                    [s.get(stat_key, np.nan) for s in worker_stats])
            else:
                stats[stat_key] = worker_stats[0][stat_key]
        return stats

    def _train_epochs(self, **params):
        remote_worker_stats = []
        for i, w in enumerate(self.remote_workers):
            stats = w.train_epochs.remote(**params)
            remote_worker_stats.append(stats)

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)
        else:
            return success, None
