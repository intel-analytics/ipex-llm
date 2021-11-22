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


KERAS_API_NAME = 'keras'

_Attributes = collections.namedtuple(
    'ExportedApiAttributes', ['names'])

# Attribute values must be unique to each API.
API_ATTRS = {
    KERAS_API_NAME: _Attributes(
        '_keras_api_names')
}


_NAME_TO_SYMBOL_MAPPING = dict()


# def get_symbol_from_name(name):
  # return _NAME_TO_SYMBOL_MAPPING.get(name)


# def get_v2_names(symbol):
#   """Get a list of TF 2.0 names for this symbol.
#
#   Args:
#     symbol: symbol to get API names for.
#
#   Returns:
#     List of all API names for this symbol including TensorFlow and
#     Estimator names.
#   """
#   names_v2 = []
#   keras_api_attr = API_ATTRS[KERAS_API_NAME].names
#
#   if not hasattr(symbol, '__dict__'):
#     return names_v2
#   if keras_api_attr in symbol.__dict__:
#     names_v2.extend(getattr(symbol, keras_api_attr))
#   return names_v2


class api_export(object):  # pylint: disable=invalid-name
  """Provides ways to export symbols to the TensorFlow API."""

  def __init__(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
        v1: Names for the TensorFlow V1 API. If not set, we will use V2 API
          names both for TensorFlow V1 and V2 APIs.
        overrides: List of symbols that this is overriding
          (those overrided api exports will be removed). Note: passing overrides
          has no effect on exporting a constant.
        api_name: Name of the API you want to generate (e.g. `tensorflow` or
          `estimator`). Default is `tensorflow`.
        allow_multiple_exports: Allow symbol to be exported multiple time under
          different names.
    """
    self._names = args
    self._api_name = kwargs.get('api_name', KERAS_API_NAME)
    # self._overrides = kwargs.get('overrides', [])
    # self._allow_multiple_exports = kwargs.get('allow_multiple_exports', False)


  def __call__(self, func):
    """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set.

    Raises:
      SymbolAlreadyExposedError: Raised when a symbol already has API names
        and kwarg `allow_multiple_exports` not set.
    """
    api_names_attr = API_ATTRS[self._api_name].names
    # Undecorate overridden names
    # for f in self._overrides:
    #   _, undecorated_f = tf_decorator.unwrap(f)
    #   delattr(undecorated_f, api_names_attr)

    _, undecorated_func = tf_decorator.unwrap(func)
    self.set_attr(undecorated_func, api_names_attr, self._names)

    for name in self._names:
      _NAME_TO_SYMBOL_MAPPING[name] = func
    return func

  def set_attr(self, func, api_names_attr, names):
    # Check for an existing api. We check if attribute name is in
    # __dict__ instead of using hasattr to verify that subclasses have
    # their own _tf_api_names as opposed to just inheriting it.
    # if api_names_attr in func.__dict__:
    #   if not self._allow_multiple_exports:
    #     raise SymbolAlreadyExposedError(
    #         'Symbol %s is already exposed as %s.' %
    #         (func.__name__, getattr(func, api_names_attr)))  # pylint: disable=protected-access
    setattr(func, api_names_attr, names)


# def kwarg_only(f):
#   """A wrapper that throws away all non-kwarg arguments."""
#   f_argspec = tf_inspect.getargspec(f)
#
#   def wrapper(*args, **kwargs):
#     if args:
#       raise TypeError(
#           '{f} only takes keyword args (possible keys: {kwargs}). '
#           'Please pass these args as kwargs instead.'
#           .format(f=f.__name__, kwargs=f_argspec.args))
#     return f(**kwargs)
#
#   return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)


keras_export = functools.partial(api_export, api_name=KERAS_API_NAME)
