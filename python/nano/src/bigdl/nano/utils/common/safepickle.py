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

import pickle
import hmac
import hashlib
from bigdl.nano.utils.common import invalidInputError

# Refer to this guide https://www.synopsys.com/blogs/software-security/python-pickling/
# To safely use python pickle


class SafePickle:
    key = b'shared-key'
    """
    Example:
        >>> from bigdl.nano.utils.common import SafePickle
        >>> with open(file_path, 'wb') as file:
        >>>     signature = SafePickle.dump(data, file, return_digest=True)
        >>> with open(file_path, 'rb') as file:
        >>>     data = SafePickle.load(file, signature)
    """
    @classmethod
    def dump(self, obj, file, return_digest=False, *args, **kwargs):
        if return_digest:
            pickled_data = pickle.dumps(obj)
            file.write(pickled_data)
            digest = hmac.new(self.key, pickled_data, hashlib.sha1).hexdigest()
            return digest
        else:
            pickle.dump(obj, file, *args, **kwargs)

    @classmethod
    def load(self, file, digest=None, *args, **kwargs):
        if digest:
            content = file.read()
            new_digest = hmac.new(self.key, content, hashlib.sha1).hexdigest()
            if digest != new_digest:
                invalidInputError(False, 'Pickle safe check failed')
            file.seek(0)
        return pickle.load(file, *args, **kwargs)
