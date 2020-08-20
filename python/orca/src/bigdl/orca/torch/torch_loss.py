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
import sys
from bigdl.nn.criterion import Criterion
from pyspark.serializers import CloudPickleSerializer
from importlib.util import find_spec

if sys.version < '3.7':
    print("WARN: detect python < 3.7, if you meet zlib not available " +
          "exception on yarn, please update your python to 3.7")

if find_spec('jep') is None:
    raise Exception("jep not found, please install jep first.")


class TorchLoss(Criterion):
    """
    TorchLoss wraps a loss function for distributed inference or training.
    This TorchLoss should be used with TorchModel.
    """

    def __init__(self, criterion_bytes, bigdl_type="float"):
        """
        :param bigdl_type:
        """
        super(TorchLoss, self).__init__(None, bigdl_type, criterion_bytes)

    @staticmethod
    def from_pytorch(criterion):
        bys = CloudPickleSerializer.dumps(CloudPickleSerializer, criterion)
        net = TorchLoss(bys)
        return net
