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
import sys
import gc
from tempfile import NamedTemporaryFile

from pyspark.broadcast import Broadcast
from pyspark.broadcast import _from_id
from bigdl.nn.layer import Model

def _from_id_and_type(bid, bigdl_type):
    result = _from_id(bid)
    return ModelBroadcast(path=result._path, bigdl_type=bigdl_type)

def broadcast_model(sc, layer):
    return ModelBroadcast(sc, layer, sc._pickled_broadcast_vars)

class ModelBroadcast(Broadcast):

    def __init__(self, sc=None, layer=None, pickle_registry=None, path=None, bigdl_type="float"):
        """
        Should not be called directly by users -- use L{SparkContext.broadcast()}
        instead.
        """
        if layer is not None:
            self.bigdl_type = layer.bigdl_type
        else:
            self.bigdl_type = bigdl_type
        super(ModelBroadcast, self).__init__(sc, layer, pickle_registry, path)

    def dump(self, value, f):
        try:
            value.saveModel(f.name, over_write=True)
        except Exception as e:
            msg = "Could not serialize broadcast: %s" % e.__class__.__name__
            if not self.sc.version.startswith("2.1"):
                from pyspark.cloudpickle import print_exec
            else:
                from pyspark.util import print_exec
            print_exec(sys.stderr)
            raise ValueError(msg)
        f.close()
        return f.name

    def _load(self, path):
        return Model.loadModel(path, bigdl_type=self.bigdl_type)

    @property
    def value(self):
        """ Return the broadcasted value
        """
        if not hasattr(self, "_value") and self._path is not None:
            self._value = self._load(self._path)
        return self._value

    def __reduce__(self):
        if self._jbroadcast is None:
            raise Exception("Broadcast can only be serialized in driver")
        self._pickle_registry.add(self)
        return _from_id_and_type, (self._jbroadcast.id(), self.bigdl_type)
