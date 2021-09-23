# +
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json


def convert_bayes_configs(config):
    selected_features = []
    new_config = {}
    for config_name, config_value in config.items():
        if config_name.startswith('bayes_feature'):
            # print(config_name, config_value)
            if config_value >= 0.5:
                feature_name = config_name.replace('bayes_feature_', '')
                selected_features.append(feature_name)
        elif config_name == 'batch_size_log':
            batch_size = int(2 ** config_value)
            new_config['batch_size'] = batch_size
        elif config_name.endswith('float'):
            int_config_name = config_name.replace('_float', '')
            int_config_value = int(config_value)
            new_config[int_config_name] = int_config_value
        else:
            new_config[config_name] = config_value
    if selected_features:
        new_config['selected_features'] = json.dumps(selected_features)
    # print("config after bayes conversion is ", new_config)
    return new_config
