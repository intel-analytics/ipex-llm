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
from random import randint
from uuid import uuid4
from bigdl.ppml.fl.psi.psi_intersection import PsiIntersection
from bigdl.ppml.fl.nn.generated.psi_service_pb2_grpc import *
from bigdl.ppml.fl.nn.generated.psi_service_pb2 import *



class PSIServiceImpl(PSIServiceServicer):
    def __init__(self, conf) -> None:
        self.client_salt = None
        self.client_secret = None
        self.client_shuffle_seed = 0
        # self.psi_collections = {}
        self.psi_intersection = PsiIntersection(conf['clientNum'])

    def getSalt(self, request, context):
        if self.client_salt is not None:
            salt = self.client_salt
        else:
            salt = str(uuid4())
            self.client_salt = salt
        
        if self.client_secret is None:
            self.client_secret = request.secure_code
        elif self.client_secret != request.secure_code:
            salt = ""

        if self.client_shuffle_seed == 0:
            self.client_shuffle_seed = randint(0, 100)
        return SaltReply(salt_reply=salt)

    def uploadSet(self, request, context):
        client_id = request.client_id
        ids = request.hashedID
        self.psi_intersection.add_collection(ids)
        logging.info(f"{len(self.psi_intersection.collection)}-th collection added")
        return UploadSetResponse(status=1)
        

    def downloadIntersection(self, request, context):
        intersection = self.psi_intersection.get_intersection()
        return DownloadIntersectionResponse(intersection=intersection)
