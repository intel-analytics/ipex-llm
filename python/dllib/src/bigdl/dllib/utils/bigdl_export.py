# This file is adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/tf_export.py
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

import collections
import functools
import sys

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

Keras_API_NAME = 'keras'

_Attributes = collections.namedtuple(
    'ExportedApiAttributes', ['names'])

API_ATTRS = {
    Keras_API_NAME: _Attributes(
        '_keras_api_names')
}
_NAME_TO_SYMBOL_MAPPING = dict()


class api_export(object):  # pylint: disable=invalid-name
    def __init__(self, *args, **kwargs):  # pylint: disable=g-doc-args
        """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
        api_name: Name of the API you want to generate (e.g. `tensorflow` or
          `estimator`). Default is `keras`.
    """
        self._names = args
        self._api_name = kwargs.get('api_name', Keras_API_NAME)

    def __call__(self, func):
        api_names_attr = API_ATTRS[self._api_name].names

        _, undecorated_func = tf_decorator.unwrap(func)
        self.set_attr(undecorated_func, api_names_attr, self._names)

        for name in self._names:
            _NAME_TO_SYMBOL_MAPPING[name] = func
            sys.modules[name] = func
        return func

    def set_attr(self, func, api_names_attr, names):
        setattr(func, api_names_attr, names)


keras_export = functools.partial(api_export, api_name=Keras_API_NAME)
