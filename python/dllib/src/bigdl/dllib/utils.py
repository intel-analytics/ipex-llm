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

if sys.version >= '3':
    long = int
    unicode = str


def toMultiShape(shape):
    if any(isinstance(i, list) for i in shape):  # multi shape
        return shape
    elif any(isinstance(i, tuple) for i in shape):
        return [list(s) for s in shape]
    elif isinstance(shape, tuple):
        return [list(shape)]
    else:
        return [shape]


# TODO: create a shape mapping here.
def remove_batch(shape):
    if any(isinstance(i, list) or isinstance(i, tuple) for i in shape):  # multi shape
        return [remove_batch(s) for s in shape]
    else:
        return list(shape[1:])
