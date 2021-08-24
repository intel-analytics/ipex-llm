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

from bigdl.dllib.utils.common import *


def save_keras_definition(keras_model, path):
    """
    Save a Keras model definition to JSON with given path
    """
    model_json = keras_model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)


def dump_keras(keras_model, json_path=None, hdf5_path=None, dump_weights=False):
    tmp_path = create_tmp_path()
    if not json_path:
        json_path = tmp_path + ".json"
    if not hdf5_path:
        hdf5_path = tmp_path + ".hdf5"
    save_keras_definition(keras_model, json_path)
    print("json path: " + json_path)
    if dump_weights:
        keras_model.save(hdf5_path)
        print("hdf5 path: " + hdf5_path)
    return json_path, hdf5_path
