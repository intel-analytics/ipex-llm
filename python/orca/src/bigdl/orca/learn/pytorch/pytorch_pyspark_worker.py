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


from subprocess import call
from pyspark import BarrierTaskContext
from bigdl.orca.learn.pytorch.torch_runner import TorchRunner
import torch.distributed as dist
import logging
from bigdl.orca.learn.utils import save_pkl
import os
import tempfile

from pyspark import BarrierTaskContext, TaskContext
from bigdl.orca.learn.utils import save_pkl, duplicate_stdout_stderr_to_file, get_rank
from bigdl.orca.learn.log_monitor import LogMonitor
from bigdl.dllib.utils.log4Error import *


logger = logging.getLogger(__name__)


class PytorchPysparkWorker(TorchRunner):
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 size,
                 cluster_info,
                 cores_per_worker,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 training_operator_cls=None,
                 config=None,
                 use_tqdm=False,
                 scheduler_step_freq=None,
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
        super().__init__(model_creator, optimizer_creator, loss_creator, metrics, scheduler_creator,
                         training_operator_cls, config, use_tqdm, scheduler_step_freq, sync_stats,
                         log_level=log_level)

        self.state_dict = state_dict
        self.size = size
        self.mode = mode
        self.backend = backend
        self.cluster_info = cluster_info
        invalidInputError(model_dir, "model_dir cannot be null")
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
        if TaskContext.get():
            partition_id = TaskContext.get().partitionId()
        else:
            partition_id = BarrierTaskContext().get().partitionId()
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
            self.setup_torch_distribute(tcp_store_host=driver_ip,
                                        tcp_store_port=driver_tcp_store_port,
                                        world_rank=self.rank,
                                        world_size=self.size)
        else:
            self.rank = 0
            self.setup_components()
            self.setup_operator(self.models)

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False,
                     info=None, wrap_dataloader=None, callbacks=None,
                     validation_data_creator=None):
        self.load_state_dict(self.state_dict.value)
        stats_list = super().train_epochs(data_creator=data_creator,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          profile=profile,
                                          info=info,
                                          wrap_dataloader=wrap_dataloader,
                                          callbacks=callbacks,
                                          validation_data_creator=validation_data_creator)
        state_dict = self.get_state_dict()

        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)

        if self.rank == 0:
            save_pkl(state_dict, os.path.join(self.model_dir, "state.pkl"))

        return [stats_list]

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 info=None, wrap_dataloader=None):
        """Evaluates the model on the validation data set."""
        self.load_state_dict(self.state_dict.value)
        validation_stats = super().validate(data_creator, batch_size, num_steps, profile, info,
                                            wrap_dataloader)
        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
        return [validation_stats]

    def predict(self, data_creator, batch_size=32, profile=False):
        """Evaluates the model on the validation data set."""
        config = self.config.copy()
        self._toggle_profiling(profile=profile)

        partition = data_creator(config, batch_size)
        self.load_state_dict(self.state_dict.value)
        result = super().predict(partition=partition, batch_size=batch_size, profile=profile)
        if self.log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
        return result

    def shutdown(self):
        """Attempts to shut down the worker."""
        dist.destroy_process_group()
        super().shutdown()
