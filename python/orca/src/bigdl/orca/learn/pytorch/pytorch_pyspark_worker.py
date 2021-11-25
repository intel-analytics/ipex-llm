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


from pyspark import BarrierTaskContext
from contextlib import closing
import socket
from bigdl.orca.learn.pytorch.torch_runner import TorchRunner
import torch.distributed as dist
import logging


def find_ip_and_port(pre_iter):
    tc = BarrierTaskContext().get()
    address = tc.getTaskInfos()[tc.partitionId()].address.split(":")[0]
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tc.barrier()
        free_ip_port = f"{address}:{s.getsockname()[1]}"
    return [free_ip_port]


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
                 log_level=logging.INFO):
        super().__init__(model_creator, optimizer_creator, loss_creator, metrics, scheduler_creator,
                         training_operator_cls, config, use_tqdm, scheduler_step_freq, sync_stats,
                         log_level=log_level)

        self.state_dict = state_dict
        self.size = size
        self.mode = mode
        self.backend = backend
        self.cluster_info = cluster_info

        self.setup(cores_per_worker)
        if self.backend == "torch-distributed":
            self.setup_distributed(self.mode, self.cluster_info)

    def setup_distributed(self, mode, cluster_info):
        if mode == "fit":
            self.rank = self._get_rank(cluster_info)
            print("cluster is: ", cluster_info)
            address = f"tcp://{cluster_info[0]}"
            self.setup_torch_distribute(url=address,
                                        world_rank=self.rank,
                                        world_size=self.size)
        else:
            self.rank = 0
            self.setup_components()
            self.setup_operator(self.models)

    @staticmethod
    def _get_rank(cluster_info):
        # As task placement may not be identical between two different jobs,
        # we cannot simply index cluster_info using partitionId to get current
        # ip and port.
        # The approach here is to first get all tasks' ip in this job and compute
        # a local rank by counting how many tasks has the same ip but with lower id.
        # We then use the local rank to find the right slot in cluster_info to find
        # the right global_rank.
        tc = BarrierTaskContext().get()
        infos = tc.getTaskInfos()
        idx = tc.partitionId()
        local_ip = infos[idx].address.split(":")[0]
        local_rank = 0
        for i in range(0, idx):
            if infos[i].address.startswith(local_ip):
                local_rank += 1
        global_rank = -1
        local_count = 0
        for node in cluster_info:
            if node.startswith(local_ip):
                local_count += 1
            global_rank += 1
            if local_count == local_rank + 1:
                break
        return global_rank

    def train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False,
                     info=None, wrap_dataloader=None):
        stats_list = super().train_epochs(data_creator, epochs, batch_size, profile, info,
                                          wrap_dataloader)
        state_dict = self.get_state_dict()

        if self.rank == 0:
            return [(state_dict, stats_list)]
        else:
            return [(None, stats_list)]

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 info=None, wrap_dataloader=None):
        """Evaluates the model on the validation data set."""
        self.load_state_dict(self.state_dict)
        validation_stats = super().validate(data_creator, batch_size, num_steps, profile, info,
                                            wrap_dataloader)
        return [validation_stats]

    def predict(self, data_creator, batch_size=32, profile=False):
        """Evaluates the model on the validation data set."""
        config = self.config.copy()
        self._toggle_profiling(profile=profile)

        partition = data_creator(config, batch_size)
        self.load_state_dict(self.state_dict)
        return super().predict(partition=partition, batch_size=batch_size, profile=profile)

    def shutdown(self):
        """Attempts to shut down the worker."""
        dist.destroy_process_group()
        super().shutdown()
