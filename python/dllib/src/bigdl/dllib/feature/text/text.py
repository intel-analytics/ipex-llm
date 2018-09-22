#
# Copyright 2018 Analytics Zoo Authors.
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

import sys
import six
from bigdl.util.common import JavaValue, callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class TextFeature(JavaValue):

    def __init__(self, text=None, label=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if jvalue:
            self.value = jvalue
        else:
            assert isinstance(text, six.string_types), "text of a TextFeature should be a string"
            if label is not None:
                self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                           text, int(label))
            else:
                self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                           text)

    def get_text(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetText", self.value)

    def get_label(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetLabel", self.value)

    def has_label(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureHasLabel", self.value)

    def set_label(self, label):
        self.value = callBigDlFunc(self.bigdl_type, "textFeatureSetLabel", self.value, int(label))
        return self

    def get_tokens(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetTokens", self.value)

    def get_sample(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetSample", self.value)

    def keys(self):
        return callBigDlFunc(self.bigdl_type, "textFeatureGetKeys", self.value)
