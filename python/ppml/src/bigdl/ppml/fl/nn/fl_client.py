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


import logging
import grpc
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, PredictRequest, UploadMetaRequest
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
import yaml
import threading
from bigdl.dllib.utils.log4Error import invalidInputError


class FLClient(object):
    channel = None
    _lock = threading.Lock()
    client_id = None
    target = "localhost:8980"
    secure = False
    creds = None

    @staticmethod
    def set_client_id(client_id: int) -> None:
        FLClient.client_id = client_id
    
    @staticmethod
    def set_target(target: str) -> None:
        FLClient.target = target

    @staticmethod
    def ensure_initialized() -> None:
        with FLClient._lock:
            if FLClient.channel == None:
                if FLClient.secure:
                    FLClient.channel = grpc.secure_channel(FLClient.target, FLClient.creds)
                else:
                    FLClient.channel = grpc.insecure_channel(FLClient.target)
    
    @staticmethod
    def load_config() -> None:
        try:
            with open('ppml-conf.yaml', 'r') as stream:
                conf = yaml.safe_load(stream)
                if 'privateKeyFilePath' in conf:
                    FLClient.secure = True
                    with open(conf['privateKeyFilePath'], 'rb') as f:
                        FLClient.creds = grpc.ssl_channel_credentials(f.read())
        except yaml.YAMLError as e:
            logging.warn('Loading config failed, using default config ')
        except Exception as e:
            logging.warn('Failed to find config file "ppml-conf.yaml", using default config')

    