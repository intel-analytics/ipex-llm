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

from bigdl.dllib.utils.common import *


def init_fl_context(bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initFLContext")

class FLClientClosable(JavaValue):
    def __init__(self, jvalue=None, bigdl_type="float", *args):
        super().__init__(jvalue, bigdl_type, *args)

    def set_fl_client(self, fl_client):
        return callBigDlFunc(self.bigdl_type, "flClientClosableSetFLClient", self.value, fl_client)

