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

import functools
import sys

Keras_API_NAME = 'keras'


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
        for name in self._names:
            sys.modules[name] = func
        return func


keras_export = functools.partial(api_export, api_name=Keras_API_NAME)
