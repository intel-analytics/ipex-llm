# Copyright 2016 The BigDL Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Tensorboardx
# MIT License

# Copyright (c) 2017 Tzu-Wei Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from ray.tune.utils import flatten_dict
import numpy as np
import os
import math

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from bigdl.dllib.utils.log4Error import *

# types will be supported natively as continuous, which can be supported in both config and metrics
VALID_SUMMARY_TYPES = (int, float)
# types will be transformed to float, which can be supported in both config and metrics
VALID_NUMPY_SUMMARY_TYPES = (np.float32, np.float64, np.int32)
# types will be transformed to str, which can be supported in config
VALID_SEQ_SUMMARY_TYPES = (list, tuple)


class TensorboardLogger:
    def __init__(self, logs_dir="", writer=None, name="AutoML"):
        '''
        Initialize a tensorboard logger

        Note that this logger relies on tensorboardx and only provide tensorboard hparams log.
        An ImportError will be raised for the lack of torch

        :param logs_dir: root directory for the log, default to the current working dir
        :param writer: shared tensorboard SummaryWriter, default to None.
        '''
        self.logs_dir = logs_dir
        self.name = name
        self._file_writer = None
        if writer:
            self._file_writer = writer
        else:
            self._file_writer = SummaryWriter(log_dir=self.logs_dir)

    def run(self, config, metric):
        '''
        Write log files(event files)

        The log files is arranged as following:
        self.logs_dir
        |--eventfile_all
        |--Trail_1
        |  |--eventfile_1
        |--Trail_2
        |  |--eventfile_2
        ...
        :param config: A dictionary. Keys are trail name,
            value is a dictionary indicates the trail config
        :param metric: A dictionary. Keys are trail name,
            value is a dictionary indicates the trail metric results for each iteration

        Example:
        Config = {"run1":{"lr":0.001, "hidden_units": 32},
                  "run2":{"lr":0.01, "hidden_units": 64}}
        Metric = {"run1":{"acc":0.91, "time": 32.13},
                  "run2":{"acc":[0.93, 0.95], "time": [61.33, 62.44]}}

        Note that the keys of config and metric should be exactly the same
        '''
        # keys check
        invalidInputError(config.keys() == metric.keys(),
                          "The keys of config and metric should be exactly the same")

        new_config = {}
        hparam_domain_discrete = {}
        for key in config.keys():
            new_config[key] = {}
            for k, value in config[key].items():
                if value is None:
                    pass
                if type(value) in VALID_SUMMARY_TYPES:
                    new_config[key][f'{self.name}/' + k] = value
                if type(value) in VALID_NUMPY_SUMMARY_TYPES and not np.isnan(value):
                    new_config[key][f'{self.name}/' + k] = float(value)
                if type(value) in VALID_SEQ_SUMMARY_TYPES:
                    new_config[key][f'{self.name}/' + k] = str(value)
                    if f'{self.name}/' + k in hparam_domain_discrete:
                        hparam_domain_discrete[f'{self.name}/' + k].add(str(value))
                    else:
                        hparam_domain_discrete[f'{self.name}/' + k] = set([str(value)])
        for k, v in hparam_domain_discrete.items():
            hparam_domain_discrete[k] = list(v)
        for key in new_config.keys():
            if new_config[key] == {}:
                del new_config[key]

        new_metric = {}
        for key in metric.keys():
            new_metric[key] = {}
            for k, value in metric[key].items():
                if not isinstance(value, list):
                    value = [value]
                if value[-1] is None:
                    continue
                if type(value[-1]) in VALID_SUMMARY_TYPES and not math.isnan(value[-1]):
                    new_metric[key][f'{self.name}/' + k] = value
                if type(value[-1]) in VALID_NUMPY_SUMMARY_TYPES and not np.isnan(value[-1]):
                    new_metric[key][f'{self.name}/' + k] = list(map(float, value))
        for key in new_metric.keys():
            if new_metric[key] == {}:
                del new_metric[key]

        # hparams log write
        for key in new_metric.keys():
            self._write_hparams(new_config[key],
                                new_metric[key],
                                name=key.replace("/", "_"),
                                hparam_domain_discrete=hparam_domain_discrete)

    def _write_hparams(self, hparam_dict, metric_dict, name, hparam_domain_discrete):
        # adapted from
        # https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/writer.py#L336-L376
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        w_hp = SummaryWriter(log_dir=os.path.join(self._file_writer.log_dir, name))
        w_hp.file_writer.add_summary(exp)
        w_hp.file_writer.add_summary(ssi)
        w_hp.file_writer.add_summary(sei)
        for k, values in metric_dict.items():
            global_step = 0
            for v in values:
                w_hp.add_scalar(k, v, global_step)
                global_step += 1
        w_hp.close()

    @staticmethod
    def _ray_tune_searcher_log_adapt(analysis):
        # config
        config = analysis.get_all_configs()
        # metric
        metric_raw = analysis.fetch_trial_dataframes()
        metric = {}
        for key, value in metric_raw.items():
            metric[key] = dict(zip(list(value.columns), list(map(list, value.values.T))))
            config[key]["address"] = key
        return config, metric

    def close(self):
        '''
        Close the logger
        '''
        self._file_writer.close()
