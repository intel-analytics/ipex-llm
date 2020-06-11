#
# Copyright 2018 Analytics Zoo Authors.
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
import subprocess
import ray.services
from dmlc_tracker.tracker import get_host_ip
from zoo.orca.learn.mxnet.mxnet_runner import MXNetRunner
from zoo.orca.learn.mxnet.utils import find_free_port


class MXNetTrainer(object):
    """
    MXNetTrainer provides an automatic setup for synchronous distributed MXNet training.

    :param config: A dictionary for training configurations. Keys must include the following:
    batch_size, optimizer, optimizer_params, log_interval.
    optimizer should be an MXNet optimizer or its string representation.
    optimizer_params should be a dict in companion with the optimizer. It can contain learning_rate
    and other optimization configurations.
    log_interval should be an integer, specifying the interval for logging throughput and metrics
    information (if any) during the training process.
    You can call create_trainer_config to create the config easily.
    You can specify "seed" in config to set random seed.
    You can specify "init" in seed to set model initializer.

    :param train_data: An instance of xShards or a function that takes config and kv as arguments
    and returns an MXNet DataIter/DataLoader for training.
    You can specify data related configurations for this function in the config argument above.
    kv is an instance of MXNet distributed key-value store. kv.num_workers and kv.rank
    can be used in this function to split data for different workers if necessary.

    :param model_creator: A function that takes config as argument and returns an MXNet model.
    The model can be defined either using MXNet symbolic API or imperative(gluon) API.

    :param loss_creator: A function that takes config as argument and returns an MXNet loss.
    This is not needed for symbolic API where loss is already defined as model output.

    :param train_resize_batch_num: The number of batches per epoch to resize to. Default is None.
    You might need to specify this if the size of train_data for each worker varies.

    :param eval_metrics_creator: A function that takes config as argument and returns one or
    a list of MXNet metrics or corresponding string representations of metrics, for example,
    'accuracy'. This is not needed if you don't need evaluation on the training data set.

    :param test_data: An instance of xShards or a function that takes config and kv as arguments
    and returns an MXNet DataIter/DataLoader for testing.
    You can specify data related configurations for this function in the config argument above.
    kv is an instance of MXNet distributed key-value store. kv.num_workers and kv.rank
    can be used in this function to split data for different workers if necessary.

    :param validation_metrics_creator: A function that takes config as argument and returns one or
    a list of MXNet metrics or corresponding string representations of metrics, for example,
    'accuracy'. This is not needed if you don't have validation data throughout the training.

    :param num_workers: The number of workers for distributed training. Default is 1.

    :param num_servers: The number of servers for distributed training. Default is None and in this
    case it would be equal to the number of workers.

    :param runner_cores: The number of CPU cores allocated for each MXNet worker and server.
    Default is None. You may need to specify this for better performance.
    """
    def __init__(self, config, train_data, model_creator,
                 loss_creator=None, train_resize_batch_num=None, eval_metrics_creator=None,
                 test_data=None, validation_metrics_creator=None,
                 num_workers=1, num_servers=None, runner_cores=None):
        self.config = config
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.validation_metrics_creator = validation_metrics_creator
        self.eval_metrics_creator = eval_metrics_creator
        self.num_workers = num_workers
        self.num_servers = num_servers if num_servers else self.num_workers
        self.train_resize_batch_num = train_resize_batch_num

        from zoo.orca.data import RayXShards, SparkXShards
        if isinstance(train_data, SparkXShards):
            train_data = train_data.repartition(self.num_workers).to_ray()
            if test_data:
                assert isinstance(test_data, SparkXShards)
                test_data = test_data.repartition(self.num_workers).to_ray()
        if isinstance(train_data, RayXShards):
            if train_data.num_partitions() != self.num_workers:
                train_data.repartition(self.num_workers)
            if test_data:
                assert isinstance(test_data, RayXShards)
                if test_data.num_partitions() != self.num_workers:
                    test_data.repartition(self.num_workers)
        self.train_data = train_data
        self.test_data = test_data

        # Generate actor class
        # Add a dummy custom resource: _mxnet_worker and _mxnet_server to diff worker from server
        # if runner_cores is specified so that we can place one worker and one server on a node
        # for better performance.
        Worker = ray.remote(num_cpus=runner_cores, resources={"_mxnet_worker": 1})(MXNetRunner) \
            if runner_cores else ray.remote(MXNetRunner)
        Server = ray.remote(num_cpus=runner_cores, resources={"_mxnet_server": 1})(MXNetRunner) \
            if runner_cores else ray.remote(MXNetRunner)

        # Start runners: workers followed by servers
        self.workers = [
            Worker.remote()
            for i in range(self.num_workers)
        ]
        self.servers = [
            Server.remote()
            for i in range(self.num_servers)
        ]

        if isinstance(self.train_data, RayXShards):
            self.workers = self.train_data.colocate_actors(self.workers)
        self.runners = self.workers + self.servers

        env = {
            "DMLC_PS_ROOT_URI": str(get_host_ip()),
            "DMLC_PS_ROOT_PORT": str(find_free_port()),
            "DMLC_NUM_SERVER": str(self.num_servers),
            "DMLC_NUM_WORKER": str(self.num_workers),
        }
        envs = []
        for i in range(self.num_workers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'worker'
            envs.append(current_env)
        for i in range(self.num_servers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'server'
            envs.append(current_env)

        env['DMLC_ROLE'] = 'scheduler'
        modified_env = os.environ.copy()
        modified_env.update(env)
        # Need to contain system env to run bash
        # TODO: Need to kill this process manually?
        subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

        ray.get([
            runner.setup_distributed.remote(envs[i], self.config,
                self.train_data,
                self.model_creator,
                self.loss_creator,
                self.validation_metrics_creator,
                self.test_data,
                self.train_resize_batch_num,
                self.eval_metrics_creator)
            for i, runner in enumerate(self.runners)
        ])

    def train(self, nb_epoch=1):
        """Trains an MXNet model for several epochs."""
        stats = ray.get([w.train.remote(nb_epoch) for w in self.runners])
        return stats

    def shutdown(self):
        """Shuts down runners and releases resources."""
        for runner in self.runners:
            runner.shutdown.remote()
            runner.__ray_terminate__.remote()

# TODO: add model save and restore
# TODO: add predict, evaluate
