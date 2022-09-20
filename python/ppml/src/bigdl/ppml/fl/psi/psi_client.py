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
import time

from bigdl.dllib.utils.log4Error import invalidOperationError
from bigdl.ppml.fl.psi.utils import to_hex_string
from ..nn.fl_client import FLClient
from bigdl.ppml.fl.nn.generated.psi_service_pb2_grpc import *
from ..nn.generated.psi_service_pb2 import DownloadIntersectionRequest, SaltRequest, UploadSetRequest

class PSI(object):
    def __init__(self) -> None:
        self.stub = PSIServiceStub(FLClient.channel)
        self.hashed_ids_to_ids = {}
    
    def get_salt(self, secure_code=""):
        return self.stub.getSalt(SaltRequest(secure_code=secure_code)).salt_reply
    
    def upload_set(self, ids, salt=""):
        # TODO: add hashing
        hashed_ids = to_hex_string(ids, salt)
        self.hashed_ids_to_ids = dict(zip(hashed_ids, ids))

        return self.stub.uploadSet(
            UploadSetRequest(client_id=FLClient.client_id, hashedID=hashed_ids))

    def download_intersection(self, max_try=100, retry=3):
        for i in range(max_try):
            intersection = self.stub.downloadIntersection(
                DownloadIntersectionRequest()).intersection
            if intersection is not None and len(intersection) != 0:
                hashed_intersection = list(intersection)
                logging.info(f"Intersection completed, size {len(intersection)}")
                intersection = [self.hashed_ids_to_ids[i] for i in hashed_intersection]
                return intersection
            else:
                logging.info(f"Got empty intersection, will retry in {retry} s... {i}/{max_try}")
                time.sleep(retry)
        invalidOperationError(False, "Max retry reached, could not get intersection, exiting.")

    def get_intersection(self, ids, secure_code="", max_try=100, retry=3):
        salt = self.get_salt(secure_code)
        self.upload_set(ids, salt)
        return self.download_intersection(max_try, retry)
