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

"""
This module imports contents from CloudPickle in a way that is compatible with the
``pickle_module`` parameter of PyTorch's model persistence function: ``torch.save``
and ``torch.load``.
TODO: remove this when PyTorch have compatible pickling APIs.
"""
from cloudpickle import *
from cloudpickle import CloudPickler as Pickler
from pickle import Unpickler
