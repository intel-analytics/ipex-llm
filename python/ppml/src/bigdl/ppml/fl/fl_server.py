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

from bigdl.ppml.fl import *


class FLServer(JavaValue):
    def __init__(self, jvalue=None, *args):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, *args)

    def build(self):
        callBigDlFunc(self.bigdl_type, "flServerBuild", self.value)

    def start(self):
        callBigDlFunc(self.bigdl_type, "flServerStart", self.value)

    def stop(self):
        callBigDlFunc(self.bigdl_type, "flServerStop", self.value)

    def set_client_num(self, client_num):
        callBigDlFunc(self.bigdl_type, "flServerSetClientNum", self.value, client_num)

    def set_port(self, port):
        callBigDlFunc(self.bigdl_type, "flServerSetPort", self.value, port)

    def block_until_shutdown(self):
        callBigDlFunc(self.bigdl_type, "flServerBlockUntilShutdown", self.value)


if __name__ == '__main__':
    fl_server = FLServer()
    fl_server.build()
    fl_server.start()
    fl_server.block_until_shutdown()
