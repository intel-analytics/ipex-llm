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

from concurrent import futures
import grpc
from bigdl.ppml.fl import *
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.pytorch.service.nn_service import NNServiceImpl


class FLServer(object):
    def __init__(self, jvalue=None, *args):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        self.port = 8980 # TODO: set from config file

    def build(self):
        add_NNServiceServicer_to_server(NNServiceImpl(), self.server)
        self.server.add_insecure_port(f'[::]:{self.port}')
        print(f'gRPC server starts listening port: {self.port}')

    def start(self):
        self.server.start()
        # self.server.wait_for_termination()

    def stop(self):
        self.server.stop(None)

    
if __name__ == '__main__':
    fl_server = FLServer()
    fl_server.build()
    fl_server.start()
    fl_server.block_until_shutdown()