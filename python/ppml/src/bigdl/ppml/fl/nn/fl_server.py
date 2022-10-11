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

from bigdl.ppml.fl.nn.nn_service import NNServiceImpl
import yaml
import click
from bigdl.ppml.fl.psi.psi_service import PSIServiceImpl
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.generated.psi_service_pb2_grpc import *
from typing import Optional

# fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
# logging.basicConfig(format=fmt, level=logging.DEBUG)

class FLServer(object):    
    def __init__(self, client_num: Optional[int]=None) -> None:
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        self.port = 8980
        self.client_num = client_num
        self.secure = False
        self.load_config()
        # a chance to overwrite client num
        if client_num is not None:
            self.conf['clientNum'] = client_num

    def set_port(self, port: int) -> None:
        self.port = port
        
    def build(self) -> None:
        add_NNServiceServicer_to_server(NNServiceImpl(conf=self.conf), self.server)
        add_PSIServiceServicer_to_server(PSIServiceImpl(conf=self.conf), self.server)
        if self.secure:
            self.server.add_secure_port(f'[::]:{self.port}', self.server_credentials)
        else:
            self.server.add_insecure_port(f'[::]:{self.port}')
        logging.info(f'gRPC server starts listening port: {self.port}')

    def start(self) -> None:
        self.server.start()
        # self.server.wait_for_termination()

    def stop(self) -> None:
        self.server.stop(None)

    def load_config(self) -> None:        
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
                self.generate_conf(conf)

        except Exception as e:
            logging.warn('Failed to load config file "ppml-conf.yaml", using default config')
            self.generate_conf({})

    def generate_conf(self, conf: dict) -> None:
        self.conf = conf
        # set default parameters if not specified in config
        if 'clientNum' not in conf.keys():
            self.conf['clientNum'] = 1

    def wait_for_termination(self):
        self.server.wait_for_termination()


@click.command()
@click.option('--client_num', default=1)
def run(client_num):
    fl_server = FLServer(client_num)
    fl_server.build()
    fl_server.start()
    fl_server.wait_for_termination()


if __name__ == '__main__':
    run()
