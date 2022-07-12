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


def init_fl_context(id, target="localhost:8980"):
    callBigDlFunc("float", "initFLContext", id, target)

class FLClientClosable(JavaValue):
    def __init__(self, jvalue=None, bigdl_type="float", *args):
        super().__init__(jvalue, bigdl_type, *args)

    def set_fl_client(self, fl_client):
        return callBigDlFunc(self.bigdl_type, "flClientClosableSetFLClient", self.value, fl_client)


import unittest
import socket
from bigdl.dllib.utils.log4Error import invalidOperationError
class FLTest(unittest.TestCase):    
    def __init__(self, methodName='FLTest') -> None:
        super().__init__(methodName)
        self.port = 8980
        self.port = self.get_available_port(self.port, self.port + 10)
        self.target = f"localhost:{self.port}"

    def update_available_port(self):
        self.port = self.get_available_port(self.port, self.port + 10)
        self.target = f"localhost:{self.port}"

    def get_available_port(self, port_start, port_end):
        def is_available(p):            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', p))            
            sock.close()
            return result != 0
        for p in range(port_start, port_end):
            if is_available(p):
                return p
            else:
                logging.info(f"port {p} is not avaible, trying another...")
        invalidOperationError(False, 
            f"can not find available port in range [{port_start}, {port_end}]")
