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

from bigdl.dllib.utils.log4Error import invalidOperationError
from ..nn.fl_client import FLClient
from bigdl.ppml.fl.nn.generated.psi_service_pb2_grpc import *
from ..nn.generated.psi_service_pb2 import DownloadIntersectionRequest, SaltRequest, UploadSetRequest

class PSI(object):
    def __init__(self) -> None:
        self.stub = PSIServiceStub(FLClient.channel)
    
    def get_salt(self, secure_code=""):
        return self.stub.getSalt(SaltRequest(secure_code=secure_code)).salt_reply
    
    def upload_set(self, ids, salt=""):
        # TODO: add hashing
        return self.stub.uploadSet(
            UploadSetRequest(client_id=FLClient.client_id, hashedID=ids))

    def download_intersection(self, max_try=100, retry=3):
        for i in range(max_try):
            intersection = self.stub.downloadIntersection(
                DownloadIntersectionRequest()).intersection
            if intersection is not None:
                intersection = list(intersection)
                logging.info(f"Intersection completed, size {len(intersection)}")
                return intersection
        invalidOperationError(False, "Max retry reached, could not get intersection, exiting.")

    def get_intersection(self, ids, secure_code="", max_try=100, retry=3):
        salt = self.stub.getSalt(SaltRequest(secure_code=secure_code)).salt_reply
        self.upload_set(ids, salt)
        return self.download_intersection(max_try, retry)
