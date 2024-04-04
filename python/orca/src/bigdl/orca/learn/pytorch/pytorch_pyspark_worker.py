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


from bigdl.orca.learn.pytorch.torch_runner import TorchRunner
import torch.distributed as dist
import logging
import os
import tempfile
import copy

from pyspark import BarrierTaskContext, TaskContext
from bigdl.orca.learn.utils import save_pkl, duplicate_stdout_stderr_to_file, get_rank,\
    get_partition_id
from bigdl.orca.learn.log_monitor import LogMonitor


logger = logging.getLogger(__name__)


class PytorchPysparkWorker(TorchRunner):
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 size,
                 cores_per_worker,
                 cluster_info=None,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 config=None,
                 state_dict=None,
                 backend="torch-distributed",
                 mode="fit",
                 sync_stats=True,
                 log_level=logging.INFO,
                 model_dir=None,
                 log_to_driver=True,
                 driver_ip=None,
                 driver_log_port=None,
                 driver_tcp_store_port=None
                 ):
        super().__init__(model_creator=model_creator,
                         optimizer_creator=optimizer_creator,
                         loss_creator=loss_creator,
                         metrics=metrics,
                         scheduler_creator=scheduler_creator,
                         config=config,
                         sync_stats=sync_stats,
                         log_level=log_level)

        self.state_dict = state_dict
        self.size = size
        self.mode = mode
        self.backend = backend
        self.model_dir = model_dir
        self.log_to_driver = log_to_driver

        self.setup(cores_per_worker)
        if self.log_to_driver:
            self.log_path, self.logger_thread, self.thread_stop = \
                PytorchPysparkWorker._start_log_monitor(driver_ip, driver_log_port)
        if self.backend == "torch-distributed":
            self.setup_distributed(self.mode, cluster_info, driver_ip, driver_tcp_store_port)

    @staticmethod
    def _start_log_monitor(driver_ip, driver_log_port):
        partition_id = get_partition_id()
        log_path = os.path.join(tempfile.gettempdir(),
                                "{}_runner.log".format(partition_id))
        duplicate_stdout_stderr_to_file(log_path)
        logger_thread, thread_stop = \
            LogMonitor.start_log_monitor(driver_ip=driver_ip,
                                         driver_port=driver_log_port,
                                         log_path=log_path,
                                         partition_id=partition_id)
        return log_path, logger_thread, thread_stop

    def setup_distributed(self, mode, cluster_info, driver_ip, driver_tcp_store_port):
        if mode == "fit":
            self.rank = get_rank(cluster_info)
            logger.info(f"cluster is: {cluster_info}")
            self.setup_components()
            self.setup_torch_distribute(tcp_store_host=driver_ip,
                                        tcp_store_port=driver_tcp_store_port,
                                        world_rank=self.rank,
                                        world_size=self.size)
        else:
            self.rank = 0
            self.setup_components()
            if self.model_creator:
                self.setup_operator(self.models)

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False,
                     wrap_dataloader=None, callbacks=None,
                     validation_data_creator=None):
        if self.state_dict:
            self.load_state_dict(self.state_dict.value)
        stats_list = super().train_epochs(data_creator=data_creator,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          profile=profile,
                                          wrap_dataloader=wrap_dataloader,
                                          callbacks=callbacks,
                                          validation_data_creator=validation_data_creator)

        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)

        if self.model_dir is not None:
            if self.rank == 0:
                state_dict = self.get_state_dict()
                save_pkl(state_dict, os.path.join(self.model_dir, "state.pkl"))
            return [stats_list]
        else:
            if self.rank == 0:
                state_dict = self.get_state_dict()
                return [state_dict, stats_list]
            else:
                return [stats_list]

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 wrap_dataloader=None, callbacks=None):
        """Evaluates the model on the validation data set."""
        self.load_state_dict(self.state_dict.value)
        validation_stats = super().validate(data_creator, batch_size, num_steps, profile,
                                            wrap_dataloader, callbacks)
        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
        return [validation_stats]

    def predict(self, data_creator, batch_size=32, profile=False, callbacks=None, output_cols=None):
        """Evaluates the model on the validation data set."""
        config = copy.copy(self.config)
        self._toggle_profiling(profile=profile)

        partition = data_creator(config, batch_size)
        self.load_state_dict(self.state_dict.value)
        result = super().predict(partition=partition, batch_size=batch_size,
                                 profile=profile, callbacks=callbacks, output_cols=output_cols)
        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
        return result

    def shutdown(self):
        """Attempts to shut down the worker."""
        dist.destroy_process_group()
        super().shutdown()
