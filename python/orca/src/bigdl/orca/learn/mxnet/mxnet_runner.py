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
import time
import logging
import subprocess
import ray.services
import mxnet as mx
import numpy as np
from mxnet import gluon
from functools import reduce
from zoo.ray.utils import to_list


class MXNetRunner(object):
    """Manages a MXNet model for training."""

    def setup_distributed(self, env, config, train_data, model_creator, loss_creator=None,
                          validation_metrics_creator=None, test_data=None, train_resize_size=None,
                          eval_metrics_creator=None):
        logging.basicConfig(level=logging.INFO)  # This can print log messages to console.
        self.logger = logging.getLogger()
        assert isinstance(config, dict), "config must be a dict"
        for param in ["batch_size", "optimizer", "optimizer_params", "log_interval"]:
            assert param in config, param + " must be specified in config"
        self.config = config
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.validation_metrics_creator = validation_metrics_creator
        self.eval_metircs_creator = eval_metrics_creator
        self.is_worker = False
        env["DMLC_NODE_HOST"] = self.get_node_ip()
        if env["DMLC_ROLE"] == "worker":
            self.is_worker = True

        if self.is_worker:
            os.environ.update(env)
            self.kv = mx.kv.create("dist_sync")
            # Set seed so that the model on each worker is initialized with the same weights
            if "seed" in self.config:
                mx.random.seed(self.config["seed"])

            from zoo.orca.data import RayXShards
            if isinstance(train_data, RayXShards):
                # Retrieve train data for RayXShards
                train_partition_data = train_data.get_partitions()[self.kv.rank].get_data()
                data, label = get_data_label(train_partition_data)
                self.train_data = mx.io.NDArrayIter(data=data, label=label,
                                                    batch_size=config["batch_size"],
                                                    shuffle=True)
                if train_resize_size is not None:
                    self.train_data = mx.io.ResizeIter(self.train_data, train_resize_size)
                # retrieve val data
                if test_data is None:
                    self.val_data = None
                else:
                    assert isinstance(test_data, RayXShards),\
                        "Test data should be an instance of RayXShards, please check your input"
                    val_partition_data = test_data.get_partitions()[self.kv.rank].get_data()
                    val_data, val_label = get_data_label(val_partition_data)
                    self.val_data = mx.io.NDArrayIter(data=val_data,
                                                      label=val_label,
                                                      batch_size=config["batch_size"],
                                                      shuffle=True)
            else:
                # data_creator
                assert callable(train_data), "Train data should be an instance of xShards or " \
                                             "a callable function, please check your input"
                self.train_data = train_data(self.config, self.kv)
                if test_data is not None:
                    assert callable(test_data), "Test data should be an instance of xShards or " \
                                                "a callable function, please check your input"
                    self.val_data = test_data(self.config, self.kv)
                else:
                    self.val_data = None

            self.model = self.model_creator(self.config)
            if self.loss_creator:
                self.loss = self.loss_creator(self.config)
            else:
                self.loss = None
            if self.eval_metircs_creator:
                self.eval_metrics = self.eval_metircs_creator(self.config)
            else:
                self.eval_metrics = None
            if self.val_data:
                assert self.validation_metrics_creator, \
                    "Metrics not defined for validation, please specify metrics_creator"
                self.val_metrics = self.validation_metrics_creator(self.config)
            else:
                self.val_metrics = None
            # For BaseModule, use symbolic API. Otherwise, use imperative API.
            # TODO: change to Estimator API?
            if not isinstance(self.model, mx.module.BaseModule):
                assert self.loss, "Loss not defined for gluon model, please specify loss_creator"
                self.trainer = gluon.Trainer(self.model.collect_params(), self.config["optimizer"],
                                             optimizer_params=self.config["optimizer_params"],
                                             kvstore=self.kv)
            else:  # Trainer is not needed for symbolic API.
                self.trainer = None
        else:  # server
            # Need to use the environment on each raylet process for the correct python environment.
            # TODO: Need to kill this process manually?
            modified_env = os.environ.copy()
            modified_env.update(env)
            # For servers, just import mxnet and no need to do anything else
            subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

    def train(self, nb_epoch=1):
        """Train the model and update the model parameters."""
        stats = dict()
        if self.is_worker:
            start_time = time.time()
            if self.trainer:  # Imperative API
                for epoch in range(nb_epoch):
                    self.train_data.reset()
                    if self.eval_metrics:
                        self.eval_metrics.reset()  # metrics will accumulate for one batch
                    batch_start_time = time.time()
                    epoch_start_time = time.time()
                    for i, batch in enumerate(self.train_data):
                        data = gluon.utils.split_and_load(
                            batch.data[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
                        label = gluon.utils.split_and_load(
                            batch.label[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
                        outputs = []
                        Ls = []
                        from mxnet import autograd as ag
                        with ag.record():
                            for x, y in zip(data, label):
                                z = self.model(x)  # forward
                                L = self.loss(z, y)
                                # store the loss and do backward on a batch for better speed
                                Ls.append(L)
                                outputs.append(z)
                            ag.backward(Ls)
                        self.trainer.step(batch.data[0].shape[0])
                        if self.eval_metrics:
                            self.eval_metrics.update(label, outputs)
                        if not (i + 1) % self.config["log_interval"]:
                            # This would be logged on driver for each worker process.
                            iteration_log = \
                                "Epoch[%d] Batch[%d]  Speed: %f samples/sec  %s=%f" \
                                % (epoch, i,
                                   self.config["batch_size"] / (time.time() - batch_start_time),
                                   "loss", Ls[0].asnumpy().mean())
                            if self.eval_metrics:
                                names, accs = self.eval_metrics.get()
                                names, accs = to_list(names), to_list(accs)
                                for name, acc in zip(names, accs):
                                    iteration_log += "  %s=%f" % (name, acc)
                            self.logger.info(iteration_log)
                        batch_start_time = time.time()
                    # Epoch time log
                    self.logger.info("[Epoch %d] time cost: %f" %
                                     (epoch, time.time() - epoch_start_time))
                    # Epoch metrics log on train data
                    if self.eval_metrics:
                        epoch_train_log = "[Epoch %d] training: " % epoch
                        names, accs = self.eval_metrics.get()
                        names, accs = to_list(names), to_list(accs)
                        for name, acc in zip(names, accs):
                            epoch_train_log += "%s=%f  " % (name, acc)
                        self.logger.info(epoch_train_log)
                    # Epoch metrics log on validation data if any:
                    if self.val_data:
                        self.val_metrics.reset()
                        self.val_data.reset()
                        for batch in self.val_data:
                            data = gluon.utils.split_and_load(
                                batch.data[0].astype("float32", copy=False),
                                ctx_list=[mx.cpu()], batch_axis=0)
                            label = gluon.utils.split_and_load(
                                batch.label[0].astype("float32", copy=False),
                                ctx_list=[mx.cpu()], batch_axis=0)
                            outputs = [self.model(X) for X in data]
                            self.val_metrics.update(label, outputs)
                        epoch_val_log = "[Epoch %d] validation: " % epoch
                        names, accs = self.val_metrics.get()
                        names, accs = to_list(names), to_list(accs)
                        for name, acc in zip(names, accs):
                            epoch_val_log += "%s=%f  " % (name, acc)
                        self.logger.info(epoch_val_log)
                    # TODO: save checkpoints
                if self.eval_metrics:
                    names, accs = self.eval_metrics.get()
                    names, accs = to_list(names), to_list(accs)
                    for name, acc in zip(names, accs):
                        stats[name] = acc
            else:  # Symbolic API
                # TODO: seems no history (i.e. validation accuracy) returned by fit?
                if "init" not in self.config:
                    from mxnet.initializer import Uniform
                    self.config["init"] = Uniform(0.01)  # This is the default value for MXNet
                if self.eval_metrics is None:
                    self.eval_metrics = 'acc'
                self.model.fit(train_data=self.train_data,
                               num_epoch=nb_epoch,
                               initializer=self.config["init"],
                               kvstore=self.kv,
                               optimizer=self.config["optimizer"],
                               optimizer_params=self.config["optimizer_params"],
                               eval_data=self.val_data,
                               eval_metric=self.eval_metrics,
                               validation_metric=self.val_metrics,
                               batch_end_callback=mx.callback.Speedometer(
                                   self.config["batch_size"], self.config["log_interval"]),
                               epoch_end_callback=None if "model" not in self.config
                               else mx.callback.do_checkpoint(self.config["model"]))
            epoch_time = time.time() - start_time
            stats["epoch_time"] = epoch_time
        return stats

    def shutdown(self):
        """Attempts to shut down the runner."""
        del self.logger
        if self.is_worker:
            del self.kv
            del self.model
            del self.train_data
            del self.val_data
            del self.trainer
            del self.loss
        # TODO: also delete downloaded data as well?

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        if "node_ip" not in self.__dict__:
            self.node_ip = ray.services.get_node_ip_address()
        return self.node_ip

    def find_free_port(self):
        """Finds a free port on the current node."""
        if "port" not in self.__dict__:
            from zoo.orca.learn.mxnet.utils import find_free_port
            self.port = find_free_port()
        return self.port


def get_data_label(partition_data):
    def combine_dict(dict1, dict2):
        return {key: np.concatenate((value, dict2[key]), axis=0)
                for (key, value) in dict1.items()}

    def combine_list(list1, list2):
        return [np.concatenate((list1[index], list2[index]), axis=0)
                for index in range(0, len(list1))]

    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]
    if isinstance(partition_data[0]['x'], dict):
        data = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), data_list)
    elif isinstance(partition_data[0]['x'], np.ndarray):
        data = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                      data_list)
    elif isinstance(partition_data[0]['x'], list):
        data = reduce(lambda list1, list2: combine_list(list1, list2), data_list)

    if isinstance(partition_data[0]['y'], dict):
        label = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), label_list)
    elif isinstance(partition_data[0]['y'], np.ndarray):
        label = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                       label_list)
    elif isinstance(partition_data[0]['y'], list):
        label = reduce(lambda list1, list2: combine_list(list1, list2), data_list)

    return data, label
