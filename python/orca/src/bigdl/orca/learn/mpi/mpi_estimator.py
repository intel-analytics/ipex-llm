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
import subprocess
import types

import cloudpickle
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from torch.utils.data import Dataset, DataLoader

from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.utils.utils import get_node_ip
from bigdl.orca.learn.mpi.mpi_runner import MPIRunner
from bigdl.orca.learn.mpi.utils import *


class MPIEstimator:
    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator,
                 metrics=None,
                 scheduler_creator=None,
                 config=None,
                 init_func=None,  # Init the distributed environment for MPI if any
                 hosts=None,
                 workers_per_node=1,
                 env=None):
        """
        Create Orca MPI Estimator
        :param model_creator: A model creator function that takes the parameter "config"
               and returns a model
        :param optimizer_creator: An optimizer creator function that has two parameters "model" and
               "config" and returns a optimizer.
        :param loss_creator: An creater function to return a loss.
               Default: None if loss computation is not needed.
        :param metrics: One or a list of validation metrics. Function(s) that computes the
               metrics between the output and target tensors are also supported.
        :param scheduler_creator: A scheduler creator function that has two parameters "optimizer"
               and "config" and returns a learning rate scheduler wrapping the optimizer.
               By default a scheduler will take effect automatically every epoch.
               Default: None if no scheduler is needed.
        :param config: A parameter config dict, that plays a role of
               configuration to create model, loss, optimizer, scheduler and data.
               Default: None if no config is needed.
        :param init_func: A function takes the parameter "config" to init the distributed
                environment for MPI if any.
        :param hosts: host information to be run distributedly.
               It can be None, 'all' or list of hostname/ip.
               If hosts is None, means it runs on single(self) node.
               If hosts is 'all', it will get executor hosts from current Spark Context.
               Default: None.
        :param workers_per_node: The number of workers on each node.
        :param env: Special environment should be passed to MPI environment.
        """
        self.dir = os.getcwd()
        self.mpi_runner = MPIRunner(hosts=hosts, processes_per_node=workers_per_node, env=env)
        with open("saved_mpi_estimator.pkl", "wb") as f:
            cloudpickle.dump((model_creator, optimizer_creator, loss_creator, metrics,
                              scheduler_creator, config, init_func), f)
        self.mpi_runner.scp_file("saved_mpi_estimator.pkl", self.dir)
        # Need to put mpi_train.py in the current directory so that the PYTHONPATH is the same.
        train_file = os.path.abspath(__file__ + "/../mpi_train.py")
        p = subprocess.Popen(["cp", train_file, self.dir])
        os.waitpid(p.pid, 0)
        self.mpi_runner.scp_file(train_file, self.dir)

    # Specify feature_cols and label_cols for Spark DataFrame data.
    # Specify train_func or validate_func for customized training and validation logic.
    # Specify train_batches and validate_batches in case of unbalance data.
    # Specify validate_steps to validate periodically. Note that validation would always be
    # triggered at the end of an epoch.
    def fit(self, data, epochs=1, batch_size=32, validation_data=None, validate_batch_size=32,
            train_func=None, validate_func=None, train_batches=None, validate_batches=None,
            validate_steps=None, feature_cols=None, label_cols=None, mpi_options=None):
        """
        Run distributed training through MPI.
        :param data: An instance of a Spark DataFrame or a function
               that takes config as argument and returns a PyTorch DataLoader for
               training.
        :param epochs: The number of epochs to train the model. Default is 1.
        :param batch_size: Batch size on each workers used for training. Default is 32.
               If your training data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param validate_batch_size: Each worker's batch size for validation. Default is 32.
               If your training data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader
        :param train_func: Specific training loop to take parameters "config", "epochs", "model",
               "train_ld", "train_batches", "optimizer", "loss", "scheduler",
               "validate_func", "valid_ld", "metrics", "validate_batches" and "validate_steps".
               Default: None to use our default training loop
        :param validate_func: Specific validate function.
               Default: None to use our default validation function.
        :param train_batches: Specify train_batches in case of unbalance data.
               Default: None to train the whole train data
        :param validate_batches: Specify validate_batches in case of unbalance data.
               Default: None to validate the whole validation data
        :param validate_steps:Specify validate_steps to validate periodically.
               Note that validation would always be triggered at the end of an epoch.
        :param feature_cols: Specify the feature column names if data is Spark Dataframe
        :param label_cols: Specify the label column names if data is Spark Dataframe
        :param mpi_options: Specify str of addition mpi options.
        :return:
        """
        if isinstance(data, DataFrame):
            invalidInputError(feature_cols is not None and label_cols is not None,
                              "feature_cols and label_cols must be provided if data is"
                              " a Spark DataFrame")
            data_rdd = data.rdd.map(convert_row(feature_cols, label_cols))
            # TODO: make object store memory configurable?
            object_store_address = self.mpi_runner.launch_plasma(object_store_memory="100g")
            # partition_id, subpartition_id, subpartition_size, object_id, node_ip
            plasma_meta = data_rdd.mapPartitionsWithIndex(
                put_to_plasma(object_store_address)).collect()
            # The following is mainly for debugging and confirmation purpose.
            train_size_map = {}
            for partition_id, subpartition_id, subpartition_size, object_id, ip in plasma_meta:
                if ip not in train_size_map:
                    train_size_map[ip] = {}
                if partition_id not in train_size_map[ip]:
                    train_size_map[ip][partition_id] = []
                train_size_map[ip][partition_id].append(subpartition_size)
            size = 0
            count = 0
            for node, meta in train_size_map.items():
                for partition_id, subpartition_size in meta.items():
                    size += sum(subpartition_size)
                    count += len(subpartition_size)
                print("Node {} has {} subpartitions and {} train records".format(node, count, size))
                size = 0
                count = 0
            data_creator = plasma_data_creator(plasma_meta, object_store_address,
                                               self.mpi_runner.processes_per_node, batch_size)
            data_rdd.unpersist()
            if validation_data:
                invalidInputError(isinstance(validation_data, DataFrame),
                                  "expect validation data to be DataFrame")
                validation_data_rdd = validation_data.rdd.map(convert_row(feature_cols, label_cols))
                validate_plasma_meta = validation_data_rdd.mapPartitionsWithIndex(
                    put_to_plasma(object_store_address)).collect()
                validate_size_map = {}
                for partition_id, subpartition_id, subpartition_size, object_id, ip in \
                        validate_plasma_meta:
                    if ip not in validate_size_map:
                        validate_size_map[ip] = {}
                    if partition_id not in validate_size_map[ip]:
                        validate_size_map[ip][partition_id] = []
                        validate_size_map[ip][partition_id].append(subpartition_size)
                size = 0
                count = 0
                for node, meta in validate_size_map.items():
                    for partition_id, subpartition_size in meta.items():
                        size += sum(subpartition_size)
                        count += len(subpartition_size)
                    print("Node {} has {} subpartitions and {} test records"
                          .format(node, count, size))
                    size = 0
                    count = 0
                validation_data_creator = plasma_data_creator(
                    validate_plasma_meta, object_store_address,
                    self.mpi_runner.processes_per_node, validate_batch_size)
                validation_data_rdd.unpersist()
            else:
                validation_data_creator = None
        else:
            invalidInputError(isinstance(data, types.FunctionType),
                              "expect data is FunctionType")
            data_creator = data
            if validation_data:
                invalidInputError(isinstance(validation_data, types.FunctionType),
                                  "expect validaton data is FunctionType")
                validation_data_creator = validation_data
            else:
                validation_data_creator = None
        if not train_func:
            train_func = train
        if validation_data_creator:
            if not validate_func:
                validate_func = validate

        with open("mpi_train_data.pkl", "wb") as f:
            cloudpickle.dump((data_creator, epochs, batch_size, validation_data_creator,
                              validate_batch_size, train_func, validate_func, train_batches,
                              validate_batches, validate_steps), f)
        self.mpi_runner.scp_file("mpi_train_data.pkl", self.dir)
        self.mpi_runner.run("{}/mpi_train.py".format(self.dir),
                            mpi_options=mpi_options,
                            pkl_path=self.dir)
        if isinstance(data, DataFrame):
            self.mpi_runner.shutdown_plasma()

    def shutdown(self):
        self.mpi_runner.shutdown_plasma()


def convert_row(feature_cols, label_cols):
    def convert_for_cols(row, cols):
        result = []
        for name in cols:
            result.append(row[name])
        if len(result) == 1:
            return result[0]
        return result

    def transform(row):
        features = convert_for_cols(row, feature_cols)
        if label_cols:
            labels = convert_for_cols(row, label_cols)
            return features, labels
        else:
            return features,

    return transform


def put_to_plasma(address):
    def f(index, iterator):
        import pyarrow.plasma as plasma
        client = plasma.connect(address)
        part_size = 1000000  # TODO: Make subpartition size configurable?
        buffer = []
        sub_index = 0
        for record in iterator:
            if len(buffer) == part_size:
                res_buffer = process_records(buffer)
                object_id = client.put(res_buffer)
                buffer = [record]
                yield index, sub_index, part_size, object_id, get_node_ip()
                sub_index += 1
            else:
                buffer.append(record)
        remain_size = len(buffer)
        if remain_size > 0:
            res_buffer = process_records(buffer)
            object_id = client.put(res_buffer)
            buffer = []
            client.disconnect()
            yield index, sub_index, remain_size, object_id, get_node_ip()
        else:
            client.disconnect()

    return f


class PlasmaNDArrayDataset(Dataset):
    def __init__(self, meta_data, object_store_address, workers_per_node=1, batch_size=1):
        import pyarrow.plasma as plasma
        self.client = plasma.connect(object_store_address)
        print("Connected to plasma")

        # All the subpartitions on this node
        all_data = [subpartition for subpartition in meta_data if subpartition[4] == get_node_ip()]
        rank = int(os.environ.get("PMI_RANK", 0))
        print("Global rank: ", rank)
        # rank = int(os.environ.get("PMIX_RANK", 0))  # For OpenMPI
        local_rank = rank % workers_per_node
        print("Local rank: ", local_rank)
        data_splits = list(chunks(all_data, len(all_data) // workers_per_node))
        worker_data = data_splits[local_rank]
        if len(data_splits) == (workers_per_node + 1):  # Can't evenly split among workers
            remain_data = data_splits[-1]
            if local_rank < len(remain_data):
                worker_data += [remain_data[local_rank]]
        self.object_ids = [subpartition[3] for subpartition in worker_data]
        self.sizes = [subpartition[2] for subpartition in worker_data]
        print("Data size for worker: ", sum(self.sizes))
        self.batch_size = batch_size
        offsets = []
        for i in self.sizes:
            if len(offsets) == 0:
                offsets.append(i)
            else:
                offsets.append(offsets[-1] + i)
        self.offsets = offsets
        self.current_index = 0  # Current index for object_id; data loaded
        self.load_from_plasma(self.current_index)

    def reset(self):
        self.current_index = 0
        self.load_from_plasma(self.current_index)

    def load_from_plasma(self, index):
        print("Loading {} of size {}".format(self.object_ids[index], self.sizes[index]))
        current_data = self.client.get(self.object_ids[index], timeout_ms=0)
        self.current_x = current_data["x"]
        self.current_y = current_data["y"]
        self.current_offset = self.offsets[index]

    def __len__(self):
        return sum(self.sizes) // self.batch_size

    def __getitem__(self, i):  # Directly get a batch
        # print("Loading batch ", i)
        if i == 0 and self.current_index != 0:
            self.reset()
        current_available_size = self.current_offset - i * self.batch_size
        x_list = []
        y_list = []
        if current_available_size < self.batch_size:
            if current_available_size != 0:
                # Add all the remaining records into this batch
                x_list.append(index(self.current_x, start=-current_available_size))
                y_list.append(index(self.current_y, start=-current_available_size))
            # Load subsequent file(s) to complete the batch
            remain_size = self.batch_size - current_available_size
            while True:
                self.current_index += 1
                self.load_from_plasma(self.current_index)
                if self.sizes[self.current_index] >= remain_size:
                    x_list.append(index(self.current_x, end=remain_size))
                    y_list.append(index(self.current_y, end=remain_size))
                    break
                else:
                    x_list.append(self.current_x)
                    y_list.append(self.current_y)
                    remain_size -= self.sizes[self.current_index]
                    if remain_size == 0:
                        break
        # The current file contains a full batch
        elif current_available_size == self.batch_size:
            x_list.append(index(self.current_x, start=-current_available_size))
            y_list.append(index(self.current_y, start=-current_available_size))
        else:
            x_list.append(index(self.current_x, start=-current_available_size,
                                end=-current_available_size + self.batch_size))
            y_list.append(index(self.current_y, start=-current_available_size,
                                end=-current_available_size + self.batch_size))

        if isinstance(self.current_x, list):
            x_np = []
            for i in range(len(self.current_x)):
                x_np.append(np.concatenate([x[i] for x in x_list]))
        else:
            x_np = np.concatenate(x_list)
        y_np = np.concatenate(y_list)
        # Can put collate_fn into train_func if necessary.
        return x_np, y_np


def plasma_data_creator(meta_data, object_store_address,
                        workers_per_node=1, batch_size=1):
    def create_plasma_dataloader(config):
        dataset = PlasmaNDArrayDataset(meta_data, object_store_address,
                                       workers_per_node, batch_size)
        # TODO: support more options
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=None,
        )
        return loader

    return create_plasma_dataloader


def train(config, epochs, model, train_ld, train_batches, optimizer, loss, scheduler,
          validate_func, valid_ld, metrics, validate_batches, validate_steps):
    import torch
    import time
    total_loss = 0
    total_samp = 0
    total_iter = 0
    total_time = 0
    previous_iteration_time = None
    step = 0
    for i in range(epochs):
        model.train()
        if config['use_ipex']:
            import intel_extension_for_pytorch as ipex
            if config['bf16']:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
            else:
                model, optimizer = ipex.optimize(model, optimizer=optimizer)
        train_iter = iter(train_ld)
        for j in range(train_batches):
            # Iterate again from the beginning if running out of batches.
            if j > 0 and j % len(train_ld) == 0:
                train_iter = iter(train_ld)
            current_time = time.time()
            if previous_iteration_time:
                iteration_time = current_time - previous_iteration_time
            else:
                iteration_time = 0
            previous_iteration_time = current_time
            x, y = next(train_iter)
            if config['bf16']:
                with torch.cpu.amp.autocast():
                    o = model(x, y)
                    l = loss(o, y)
                    l_np = l.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()
            else:
                o = model(x, y)
                l = loss(o, y)
                l_np = l.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            batch_samples = y_np.shape[0]
            total_time += iteration_time
            total_loss += l_np * batch_samples
            total_iter += 1
            total_samp += batch_samples
            step += 1
            should_print = ("print_freq" in config and (step % config["print_freq"] == 0)) \
                or (j + 1 == train_batches)
            if should_print:
                average_batch_time = 1000.0 * total_time / total_iter
                total_time = 0
                average_loss = total_loss / total_samp
                total_loss = 0

                print(
                    "Finished training it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                        j + 1, train_batches, i, average_batch_time)
                    + "loss {:.6f}, ".format(average_loss)
                )
                total_iter = 0
                total_samp = 0

            should_validate = valid_ld and ((validate_steps > 0 and (step % validate_steps == 0))
                                            or (j + 1 == train_batches))
            if should_validate:
                validate_func(config, model, valid_ld, metrics, validate_batches)


# TODO: add loss
def validate(config, model, valid_ld, metrics, validate_batches):
    import torch
    from bigdl.orca.learn.metrics import Metric

    model.eval()
    metrics = Metric.convert_metrics_dict(metrics, backend="pytorch")
    valid_iter = iter(valid_ld)
    with torch.no_grad():
        for j in range(validate_batches):
            # Iterate again from the beginning if running out of batches.
            if j > 0 and j % len(valid_ld) == 0:
                valid_iter = iter(valid_ld)
            x, y = next(valid_iter)
            o = model(x, y)
            for metric in metrics.values():
                metric(o, y)
    result = {name: metric.compute() for name, metric in metrics.items()}
    output = "Validation results: "
    for metric, value in result.items():
        output += "{}:{} ".format(metric, value)
    print(output)
    return result
