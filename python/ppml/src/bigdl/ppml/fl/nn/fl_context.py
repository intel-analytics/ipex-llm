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

import threading
from bigdl.ppml.fl.nn.fl_client import FLClient


# NN FLContext
# TODO: merge FGBoost
class FLContext():
    fl_client = None
    _lock = threading.Lock()
    @staticmethod
    def init_fl_context(client_id):
        """
        Make sure a global FLClient exists, will create one if not
        """
        with FLContext._lock:
            if FLContext.fl_client is None:
                FLContext.fl_client = FLClient(client_id)
