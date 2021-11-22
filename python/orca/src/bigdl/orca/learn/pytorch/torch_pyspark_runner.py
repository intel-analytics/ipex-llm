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

# This file is adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/torch_runner.py


from pyspark import BarrierTaskContext
from pyspark.context import SparkContext
from contextlib import closing
import socket
from bigdl.orca.learn.pytorch.torch_runner import TorchRunner


def find_free_port(tc):
    address = tc.getTaskInfos()[tc.partitionId()].address.split(":")[0]
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tc.barrier()
        return f"{address}:{s.getsockname()[1]}"


def find_ip_and_port(pre_iter):
    tc = BarrierTaskContext().get()
    free_port = find_free_port(tc)
    return [free_port]


class TorchPysparkRunner(TorchRunner):
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
                 mode="fit"):
        super().__init__(model_creator, optimizer_creator, loss_creator, metrics, scheduler_creator,
                         training_operator_cls, config, use_tqdm, scheduler_step_freq)

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

    def _get_rank(self, cluster_info):
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
            return []

    def validate(self, data_creator, batch_size=32, num_steps=None, profile=False,
                 info=None, wrap_dataloader=None):
        """Evaluates the model on the validation data set."""
        validation_stats = super().validate(data_creator, batch_size, num_steps, profile, info,
                                            wrap_dataloader)
        if self.rank == 0:
            return [validation_stats]
        else:
            return []

    def predict(self, data_creator, batch_size=32, profile=False):
        """Evaluates the model on the validation data set."""
        config = self.config.copy()
        self._toggle_profiling(profile=profile)

        partition = data_creator(config, batch_size)
        return super().predict(partition=partition, batch_size=batch_size, profile=profile)

    def get_state_dict(self):
        """Returns the state of the runner."""
        state = {
            "epoch": self.epochs,
            "operator": self.training_operator.state_dict(),
            "models": [model.state_dict() for model in self.models],
            "optimizers": [opt.state_dict() for opt in self.optimizers]
        }
        if self.schedulers:
            state.update({
                "schedulers": [
                    scheduler.state_dict() for scheduler in self.schedulers
                ]
            })
        return state

    def load_state_dict(self, state):
        """Sets the state of the model."""
        for model, state_dict in zip(self.models, state["models"]):
            model.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(state_dict)
        if self.schedulers:
            for scheduler, state_dict in zip(self.schedulers,
                                             state["schedulers"]):
                scheduler.load_state_dict(state_dict)

        self.epochs = state["epoch"]
        self.training_operator.load_state_dict(state["operator"])

    def get_state_stream(self):
        """Returns a bytes object for the state dict."""
        state_dict = self.get_state_dict()
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        return _buffer.getvalue()

    def load_state_stream(self, byte_obj):
        """Loads a bytes object the training state dict."""
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer)
        return self.load_state_dict(state_dict)

    def apply(self, fn):
        return fn()

    def apply_operator(self, fn):
        return fn(self.training_operator)

    def shutdown(self):
        """Attempts to shut down the worker."""
        dist.destroy_process_group()
        del self.training_operator
        del self.validation_loader
        del self.train_loader
        del self.criterion
        del self.optimizers
        del self.models

    @property
    def given_models(self):
        if len(self.models) > 1:
            return self.models
        else:
            return self.models[0]

    @property
    def given_optimizers(self):
        if len(self.optimizers) > 1:
            return self.optimizers
        else:
            return self.optimizers[0]

    @property
    def given_schedulers(self):
        if not self.schedulers:
            return self.schedulers
        if len(self.schedulers) > 1:
            return self.schedulers
        else:
            return self.schedulers[0]
