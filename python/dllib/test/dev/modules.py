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

# Adopt from Spark and it might be refactored in the future

from functools import total_ordering

all_modules = []


@total_ordering
class Module(object):

    def __init__(self, name, python_test_goals=()):
        """
        Define a new module.

        :param name: A short module name, for display in logging and error messagesl
        :param python_test_goals: A set of Python test goals for testing this module.
        """
        self.name = name
        self.python_test_goals = python_test_goals

        all_modules.append(self)

    def __repr__(self):
        return "Module<%s>" % self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not (self.name == other.name)

    def __hash__(self):
        return hash(self.name)


bigdl_layer = Module(
    name="bigdl_layer",
    python_test_goals=[
        "bigdl.nn.layer"
    ])

bigdl_layer = Module(
    name="bigdl_criterion",
    python_test_goals=[
        "bigdl.nn.criterion"
    ])

bigdl_layer = Module(
    name="bigdl_common",
    python_test_goals=[
        "bigdl.util.common"
    ])

bigdl_optimizer = Module(
    name="bigdl_optimizer",
    python_test_goals=[
        "bigdl.optim.optimizer",
    ]
)

test_simple_integration_test = Module(
    name="simple_integration_test",
    python_test_goals=[
        "test.simple_integration_test"
    ]
)

test_load_caffe = Module(
    name="load_caffe_test",
    python_test_goals=[
        "test.load_caffe_test"
    ]
)
