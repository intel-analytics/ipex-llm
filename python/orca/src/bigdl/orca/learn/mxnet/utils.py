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

import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def create_config(optimizer="sgd", optimizer_params=None,
                  log_interval=10, seed=None, extra_config=None):
    if not optimizer_params:
        optimizer_params = {'learning_rate': 0.01}
    config = {
        "optimizer": optimizer,
        "optimizer_params": optimizer_params,
        "log_interval": log_interval,
    }
    if seed:
        config["seed"] = seed
    if extra_config:
        assert isinstance(extra_config, dict), "extra_config must be a dict"
        config.update(extra_config)
    return config
