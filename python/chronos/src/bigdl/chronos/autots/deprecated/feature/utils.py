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
import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    convert numpy array to list for JSON serialize
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_config(file_path, config, replace=False):
    """
    :param file_path: the file path of config to be saved.
    :param config: dict. The config to be saved
    :param replace: whether to replace if the config file already existed.
    :return:
    """
    if os.path.isfile(file_path) and not replace:
        with open(file_path, "r") as input_file:
            old_config = json.load(input_file)
        old_config.update(config)
        config = old_config.copy()

    file_dirname = os.path.dirname(os.path.abspath(file_path))
    if file_dirname and not os.path.exists(file_dirname):
        os.makedirs(file_dirname)

    with open(file_path, "w") as output_file:
        json.dump(config, output_file, cls=NumpyEncoder)


def save(file_path, feature_transformers=None, model=None, config=None):
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    config_path = os.path.join(file_path, "config.json")
    model_path = os.path.join(file_path, "weights_tune.h5")
    if feature_transformers is not None:
        feature_transformers.save(config_path, replace=True)
    if model is not None:
        model.save(model_path, config_path)
    if config is not None:
        save_config(config_path, config)


def load_config(file_path):
    with open(file_path, "r") as input_file:
        data = json.load(input_file)
    return data


def restore(file, feature_transformers=None, model=None, config=None):
    model_path = os.path.join(file, "weights_tune.h5")
    config_path = os.path.join(file, "config.json")
    local_config = load_config(config_path)
    if config is not None:
        all_config = config.copy()
        all_config.update(local_config)
    else:
        all_config = local_config
    if model:
        model.restore(model_path, **all_config)
    if feature_transformers:
        feature_transformers.restore(**all_config)
    return all_config
