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

import torch
import torch.nn as nn
import warnings
import random
import numpy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import multiprocessing as mp
import warnings
import math

__all__ = ['loader_to_creator',
           'np_to_creator',
           'xshard_to_np',
           'np_to_xshard',
           'set_pytorch_seed',
           'check_data',
           'np_to_dataloader',
           'read_csv',
           'delete_folder',
           'is_main_process',
           'xshard_expand_dim',
           'get_exported_module',
           'set_pytorch_thread']


def loader_to_creator(loader):
    # Warning, this data creator will not respect the batch_size changing.
    def data_creator(config, batch_size):
            return loader
    return data_creator


def np_to_creator(data):
    def data_creator(config, batch_size):
            return DataLoader(TensorDataset(torch.from_numpy(data[0]).float(),
                                            torch.from_numpy(data[1]).float()),
                              batch_size=batch_size,
                              shuffle=True)
    return data_creator


def set_pytorch_seed(seed):
    if seed is not None and isinstance(seed, int):
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)


def xshard_to_np(shard, mode="fit", expand_dim=None):
    if mode == "fit":
        data_local = shard.collect()
        return (np.concatenate([data_local[i]['x'] for i
                                in range(len(data_local))], axis=0),
                np.concatenate([data_local[i]['y'] for i
                                in range(len(data_local))], axis=0))
    if mode == "predict":
        data_local = shard.collect()
        return np.concatenate([data_local[i]['x'] for i
                               in range(len(data_local))], axis=0)
    if mode == "yhat":
        yhat = shard.collect()
        yhat = np.concatenate([yhat[i]['prediction'] for i in range(len(yhat))], axis=0)
        if len(expand_dim) >= 1:
            yhat = np.expand_dims(yhat, axis=expand_dim)
        return yhat


def xshard_expand_dim(yhat, expand_dim=None):
    if len(expand_dim) >= 1:
        yhat["prediction"] = np.expand_dims(yhat["prediction"], axis=expand_dim)
    return yhat


def np_to_xshard(x, workers_num, prefix="x"):
    from bigdl.orca.data import XShards
    x = XShards.partition(x)
    if math.floor(math.sqrt(math.sqrt(x.num_partitions()))) > workers_num:
        warnings.warn("Too many partitions and too few workers will reduce inference performance, "
                      "we recommend setting 'workers_per_node' to a large number and",
                      "not larger than the number of partitions.", category=RuntimeWarning)

    def transform_to_dict(train_data):
        return {prefix: train_data}
    return x.transform_shard(transform_to_dict)


def check_data(x, y, data_config):
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(data_config["past_seq_len"] == x.shape[-2],
                      "The x shape should be (batch_size, past_seq_len, input_feature_num), "
                      "Got past_seq_len of {} in config while x input shape of {}."
                      .format(data_config["past_seq_len"], x.shape[-2]))
    invalidInputError(data_config["future_seq_len"] == y.shape[-2],
                      "The y shape should be (batch_size, future_seq_len, output_feature_num),"
                      " Got future_seq_len of {} in config while y input shape of {}."
                      .format(data_config["future_seq_len"], y.shape[-2]))
    invalidInputError(data_config["input_feature_num"] == x.shape[-1],
                      "The x shape should be (batch_size, past_seq_len, input_feature_num),"
                      " Got input_feature_num of {} in config while x input shape of {}."
                      .format(data_config["input_feature_num"], x.shape[-1]))
    invalidInputError(data_config["output_feature_num"] == y.shape[-1],
                      "The y shape should be (batch_size, future_seq_len, output_feature_num),"
                      " Got output_feature_num of {} in config while y input shape of {}."
                      .format(data_config["output_feature_num"], y.shape[-1]))


def check_transformer_data(x, y, x_enc, y_enc, data_config):
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(data_config["past_seq_len"] == x.shape[-2],
                      "The x shape should be (batch_size, past_seq_len, input_feature_num), "
                      "Got past_seq_len of {} in config while x input shape of {}."
                      .format(data_config["past_seq_len"], x.shape[-2]))
    invalidInputError(data_config["future_seq_len"] + data_config["label_len"] == y.shape[-2],
                      "The y shape should be (batch_size, label_len + future_seq_len, "
                      "output_feature_num), Got future_seq_len plus label_len of {} "
                      "in config while y input shape of {}.".format(
                      data_config["future_seq_len"] + data_config["label_len"], y.shape[-2]))
    invalidInputError(data_config["input_feature_num"] == x.shape[-1],
                      "The x shape should be (batch_size, past_seq_len, input_feature_num),"
                      " Got input_feature_num of {} in config while x input shape of {}."
                      .format(data_config["input_feature_num"], x.shape[-1]))
    invalidInputError(data_config["output_feature_num"] == y.shape[-1],
                      "The y shape should be (batch_size, label_len + future_seq_len, "
                      "output_feature_num), Got output_feature_num of {} in config while "
                      "y input shape of {}.".format(data_config["output_feature_num"],
                                                    y.shape[-1]))
    invalidInputError(data_config["past_seq_len"] == x_enc.shape[-2],
                      "The x shape should be (batch_size, past_seq_len, time_feature_num), "
                      "Got past_seq_len of {} in config while x_enc input shape of {}."
                      .format(data_config["past_seq_len"], x_enc.shape[-2]))
    invalidInputError(data_config["future_seq_len"] + data_config["label_len"] == y_enc.shape[-2],
                      "The y shape should be (batch_size, label_len + future_seq_len, "
                      "time_feature_num), Got future_seq_len plus labnel_len of {} in config while "
                      "y_enc input shape of {}.".format(data_config["future_seq_len"] +
                                                        data_config["label_len"], y_enc.shape[-2]))


def np_to_dataloader(data, batch_size, num_processes):
    if batch_size % num_processes != 0:
        warnings.warn("'batch_size' cannot be divided with no remainder by "
                      "'self.num_processes'. We got 'batch_size' = {} and "
                      "'self.num_processes' = {}".
                      format(batch_size, num_processes))
    return DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                    torch.from_numpy(data[1])),
                      batch_size=max(1, batch_size//num_processes),
                      shuffle=True)


def read_csv(filename, loss_name='val/loss'):
    import codecs
    import csv
    fit_out = {}
    with codecs.open(filename, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row[loss_name]:
                fit_out[row['epoch']] = {'val_loss': row[loss_name]}
    return fit_out


def delete_folder(path):
    import shutil
    shutil.rmtree(path)


def is_main_process():
    return mp.current_process().name == "MainProcess"


class ExportForecastingPipeline(nn.Module):
    def __init__(self, preprocess: nn.Module,
                 inference: nn.Module, postprocess: nn.Module) -> None:
        super().__init__()
        self.preprocess = preprocess
        self.inference = inference
        self.postprocess = postprocess

    def forward(self, data):
        preprocess_output = self.preprocess(data)
        inference_output = self.inference(preprocess_output)
        postprocess_output = self.postprocess(inference_output)
        return postprocess_output


def get_exported_module(tsdata, forecaster_path, drop_dt_col):
    from bigdl.chronos.data.utils.export_torchscript \
        import get_processing_module_instance, get_index

    if drop_dt_col:
        tsdata.df.drop(columns=tsdata.dt_col, inplace=True)

    id_index, target_feature_index = get_index(tsdata.df, tsdata.id_col,
                                               tsdata.target_col, tsdata.feature_col)

    preprocess = get_processing_module_instance(tsdata.scaler, tsdata.lookback,
                                                id_index, target_feature_index,
                                                tsdata.scaler_index, "preprocessing")

    postprocess = get_processing_module_instance(tsdata.scaler, tsdata.lookback,
                                                 id_index, target_feature_index,
                                                 tsdata.scaler_index, "postprocessing")

    inference = torch.jit.load(forecaster_path)

    return torch.jit.script(ExportForecastingPipeline(preprocess, inference, postprocess))


def set_pytorch_thread(optimized_model_thread_num, thread_num):
    # optimized_model_thread_num is None means no limit is set, just keep current thread num
    if optimized_model_thread_num is not None and optimized_model_thread_num != thread_num:
        thread_num = optimized_model_thread_num
        torch.set_num_threads(thread_num)
    return thread_num
