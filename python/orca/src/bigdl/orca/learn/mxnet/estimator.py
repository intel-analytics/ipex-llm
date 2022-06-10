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
import types
import itertools
import subprocess
import ray
from dmlc_tracker.tracker import get_host_ip

from bigdl.orca.data.utils import ray_partitions_get_data_label, process_spark_xshards
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.learn.mxnet.mxnet_runner import MXNetRunner
from bigdl.orca.learn.mxnet.utils import find_free_port
from bigdl.orca.learn.ray_estimator import Estimator as OrcaRayEstimator
from bigdl.dllib.utils.log4Error import *


def partition_refs_to_creator(partition_refs, shuffle=False):

    def data_creator(config, kv):
        import mxnet as mx
        invalidInputError("batch_size" in config,
                          "batch_size must be set in config")
        data, label = ray_partitions_get_data_label(ray.get(partition_refs),
                                                    allow_tuple=False,
                                                    allow_list=False)

        train_data_iter = mx.io.NDArrayIter(data=data, label=label,
                                            batch_size=config["batch_size"],
                                            shuffle=shuffle)
        if "train_resize_batch_num" in config:
            train_data_iter = mx.io.ResizeIter(train_data_iter,
                                               config["train_resize_batch_num"])
        return train_data_iter

    return data_creator


class Estimator(object):
    @staticmethod
    def from_mxnet(*, config, model_creator, loss_creator=None, eval_metrics_creator=None,
                   validation_metrics_creator=None, num_workers=None, num_servers=None,
                   runner_cores=None):
        """
        MXNet Estimator provides an automatic setup for synchronous distributed MXNet training.

        :param config: A dictionary for training configurations. Keys must include the following:
        optimizer, optimizer_params, log_interval.
        optimizer should be an MXNet optimizer or its string representation.
        optimizer_params should be a dict in companion with the optimizer. It can contain
        learning_rate and other optimization configurations.
        log_interval should be an integer, specifying the interval for logging throughput and
        metrics information (if any) during the training process.
        You can call create_config to directly create it.
        You can specify "seed" in config to set random seed for weight initialization.
        You can specify "init" in extra_config to set model initializer for gluon models.
        :param model_creator: A function that takes config as argument and returns an MXNet model.
        The model can be defined either using MXNet symbolic API or imperative(gluon) API.
        :param loss_creator: A function that takes config as argument and returns an MXNet loss.
        This is not needed for symbolic API where loss is already defined as model output.
        :param eval_metrics_creator: A function that takes config as argument and returns one or
        a list of MXNet metrics or corresponding string representations of metrics, for example,
        'accuracy'. This is not needed if you don't need evaluation on the training data set.
        :param validation_metrics_creator: A function that takes config as argument and returns one
        or a list of MXNet metrics or corresponding string representations of metrics, for example,
        'accuracy'. This is not needed if you don't have validation data throughout the training.
        :param num_workers: The number of workers for distributed training. Default to be the number
        of nodes in the cluster.
        :param num_servers: The number of servers for distributed training. Default is None and in
        this case it would be equal to the number of workers.
        :param runner_cores: The number of CPU cores allocated for each MXNet worker and server.
        Default is None. You may need to specify this for better performance when you run in
        cluster.
        :return: MXNet Estimator object.
        """
        return MXNetEstimator(config=config, model_creator=model_creator,
                              loss_creator=loss_creator,
                              eval_metrics_creator=eval_metrics_creator,
                              validation_metrics_creator=validation_metrics_creator,
                              num_workers=num_workers, num_servers=num_servers,
                              runner_cores=runner_cores)


class MXNetEstimator(OrcaRayEstimator):
    def __init__(self, config, model_creator, loss_creator=None,
                 eval_metrics_creator=None, validation_metrics_creator=None,
                 num_workers=None, num_servers=None, runner_cores=None):
        ray_ctx = OrcaRayContext.get()
        if not num_workers:
            num_workers = ray_ctx.num_ray_nodes
        self.config = {} if config is None else config
        invalidInputError(isinstance(config, dict), "config must be a dict")
        for param in ["optimizer", "optimizer_params", "log_interval"]:
            invalidInputError(param in config, param + " must be specified in config")
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.validation_metrics_creator = validation_metrics_creator
        self.eval_metrics_creator = eval_metrics_creator
        self.num_workers = num_workers
        self.num_servers = num_servers if num_servers else self.num_workers

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
                self.model_creator,
                self.loss_creator,
                self.validation_metrics_creator,
                self.eval_metrics_creator)
            for i, runner in enumerate(self.runners)
        ])

    def fit(self, data, epochs=1, batch_size=32, validation_data=None, train_resize_batch_num=None):
        """
        Trains an MXNet model given train_data (with val_data) for several epochs.

        :param data: An instance of SparkXShards or a function that takes config and kv as
        arguments and returns an MXNet DataIter/DataLoader for training.
        You can specify data related configurations for this function in the config argument above.
        kv is an instance of MXNet distributed key-value store. kv.num_workers and kv.rank
        can be used in this function to split data for different workers if necessary.
        :param epochs: The number of epochs to train the MXNet model. Default is 1.
        :param batch_size: The number of samples per batch for each worker. Default is 32.
        :param validation_data: An instance of SparkXShards or a function that takes config and
        kv as arguments and returns an MXNet DataIter/DataLoader for validation.
        You can specify data related configurations for this function in the config argument above.
        kv is an instance of MXNet distributed key-value store. kv.num_workers and kv.rank
        can be used in this function to split data for different workers if necessary.
        :param train_resize_batch_num: The number of batches per epoch to resize to.
        Default is None. You might need to specify this if the size of train_data for each
        worker varies. MXNet distributed training would crash when the first worker finishes
        the training if the workers have unbalanced training data.
        See this issue for more details: https://github.com/apache/incubator-mxnet/issues/17651
        """
        if validation_data:
            invalidInputError(self.validation_metrics_creator,
                              "Metrics not defined for validation, please specify"
                              " validation_metrics_creator when creating the Estimator")
        from bigdl.orca.data import SparkXShards
        if isinstance(data, SparkXShards):

            ray_xshards = process_spark_xshards(data, self.num_workers)

            if validation_data is None:
                def transform_func(worker, partition_refs):
                    data_creator = partition_refs_to_creator(partition_refs, shuffle=True)

                    return worker.train.remote(data_creator,
                                               epochs,
                                               batch_size,
                                               None,
                                               train_resize_batch_num)

                worker_stats = ray_xshards.reduce_partitions_for_actors(self.workers,
                                                                        transform_func)
            else:
                val_ray_xshards = process_spark_xshards(validation_data, self.num_workers)

                def zip_func(worker, this_partition_refs, that_partition_refs):
                    data_creator = partition_refs_to_creator(this_partition_refs,
                                                             shuffle=True)
                    validation_data_creator = partition_refs_to_creator(that_partition_refs,
                                                                        shuffle=True)
                    return worker.train.remote(data_creator,
                                               epochs,
                                               batch_size,
                                               validation_data_creator,
                                               train_resize_batch_num)
                worker_stats = ray_xshards.zip_reduce_shards_with_actors(val_ray_xshards,
                                                                         self.workers,
                                                                         zip_func)
            server_stats = [server.train.remote(None, epochs, batch_size,
                                                None, train_resize_batch_num)
                            for server in self.servers]
            server_stats = ray.get(server_stats)
            server_stats = list(itertools.chain.from_iterable(server_stats))
            stats = worker_stats + server_stats

        else:  # data_creator functions; should return Iter or DataLoader
            invalidInputError(isinstance(data, types.FunctionType),
                              "train_data should be either an instance of SparkXShards or"
                              " a callable function")
            train_data_list = [data] * self.num_workers
            if validation_data:
                invalidInputError(isinstance(validation_data, types.FunctionType),
                                  "val_data should be either an instance of SparkXShards"
                                  " or a callable function")
            val_data_list = [validation_data] * self.num_workers
            self.runners = self.workers + self.servers
            # For servers, data is not used and thus just input a None value.
            train_data_list += [None] * self.num_servers
            val_data_list += [None] * self.num_servers

            stats = ray.get(
                [runner.train.remote(train_data_list[i], epochs, batch_size,
                                     val_data_list[i], train_resize_batch_num)
                 for i, runner in enumerate(self.runners)])
            stats = list(itertools.chain.from_iterable(stats))
        return stats

    def shutdown(self):
        """
        Shuts down runners and releases resources.
        """
        for runner in self.runners:
            runner.shutdown.remote()
            runner.__ray_terminate__.remote()

    def predict(self, data, batch_size=32):
        """
        Predict is not supported in MXNetEstimator
        """
        invalidInputError(False, "not implemented")

    def evaluate(self, data, batch_size=32, num_steps=None):
        """
        Evaluate is not supported in MXNetEstimator
        """
        invalidInputError(False, "not implemented")

    def get_model(self):
        """
        Get_model is not supported in MXNetEstimator
        """
        invalidInputError(False, "not implemented")

    def save(self, checkpoint):
        """
        Save is not supported in MXNetEstimator
        """
        invalidInputError(False, "not implemented")

    def load(self, checkpoint):
        """
        Load is not supported in MXNetEstimator
        """
        invalidInputError(False, "not implemented")

# TODO: add model save and restore
# TODO: add predict, evaluate
