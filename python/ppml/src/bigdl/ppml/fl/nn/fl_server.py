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
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.nn_service import NNServiceImpl
import yaml
import logging


class FLServer(object):
    def __init__(self, client_num=1):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        self.port = 8980
        self.client_num = client_num
        self.secure = False
        self.load_config()

    def set_port(self, port):
        self.port = port
        
    def build(self):
        add_NNServiceServicer_to_server(
            NNServiceImpl(client_num=self.client_num),            
            self.server)
        if self.secure:
            self.server.add_secure_port(f'[::]:{self.port}', self.server_credentials)
        else:
            self.server.add_insecure_port(f'[::]:{self.port}')
        logging.info(f'gRPC server starts listening port: {self.port}')

    def start(self):
        self.server.start()
        # self.server.wait_for_termination()

    def stop(self):
        self.server.stop(None)

    def load_config(self):        
        try:
            with open('ppml-conf.yaml', 'r') as stream:
                conf = yaml.safe_load(stream)
                if 'privateKeyFilePath' in conf:
                    self.secure = True
                    with open(conf['privateKeyFilePath'], 'rb') as f:
                        private_key = f.read()
                    with open(conf['certChainFilePath'], 'rb') as f:
                        certificate_chain = f.read()
                    self.server_credentials = grpc.ssl_server_credentials(
                         ( (private_key, certificate_chain), ) )
                if 'serverPort' in conf:
                    self.port = conf['serverPort']
                if 'clientNum' in conf:
                    self.client_num = conf['clientNum']

        except yaml.YAMLError as e:
            logging.warn('Loading config failed, using default config ')
        except Exception as e:
            logging.warn('Failed to find config file "ppml-conf.yaml", using default config')

    def wait_for_termination(self):
        self.server.wait_for_termination()

    
if __name__ == '__main__':
    fl_server = FLServer(2)
    fl_server.build()
    fl_server.start()
    fl_server.wait_for_termination()