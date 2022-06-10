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
import six
from bigdl.dllib.utils.log4Error import *


def flatten(seq):
    if isinstance(seq, list):
        results = []
        for item in seq:
            results.extend(flatten(item))
        return results

    if isinstance(seq, tuple):
        seq = list(seq)
        results = []
        for item in seq:
            results.extend(flatten(item))
        return results

    if isinstance(seq, dict):
        sorted_keys = sorted(seq.keys())
        result = []
        for key in sorted_keys:
            result.extend(flatten(seq[key]))
        return result

    return [seq]


def ptensor_to_numpy(seq):
    return [t.data.numpy() for t in flatten(seq)]


def pack_sequence_as(structure, flat_sequence):
    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)


def _yield_value(iterable):
    if isinstance(iterable, dict):
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        for value in iterable:
            yield value


def _sequence_like(instance, args):
    if isinstance(instance, dict):
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key]) for key in six.iterkeys(instance))
    else:
        # Not a namedtuple
        return type(instance)(args)


def _packed_nest_with_indices(structure, flat, index):
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def _get_attrs_values(obj):
    attrs = getattr(obj.__class__, "__attrs_attrs__")
    return [getattr(obj, a.name) for a in attrs]


def _sorted(dict_):
    try:
        return sorted(six.iterkeys(dict_))
    except TypeError:
        invalidInputError(False, "nest only supports dicts with sortable keys.")


def is_sequence(s):
    return isinstance(s, dict) or isinstance(s, list) or isinstance(s, tuple)
