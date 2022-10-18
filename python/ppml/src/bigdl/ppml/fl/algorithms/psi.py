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

from bigdl.dllib.utils.common import JavaValue


def set_psi_salt(psi_salt):
    callBigDlFunc("float", "setPsiSalt", psi_salt)


class PSI(JavaValue):
    def __init__(self, jvalue=None, *args):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, *args)

    def get_salt(self, secure_code=""):
        return callBigDlFunc(self.bigdl_type, "psiGetSalt", self.value, secure_code)

    def upload_set(self, ids, salt):
        callBigDlFunc(self.bigdl_type, "psiUploadSet", self.value, ids, salt)

    def download_intersection(self, max_try=100, retry=3000):
        return callBigDlFunc(self.bigdl_type, "psiDownloadIntersection", self.value, max_try, retry)

    def get_intersection(self, ids, max_try=100, retry=3000):
        return callBigDlFunc(self.bigdl_type, "psiGetIntersection", \
            self.value, ids, max_try, retry)
