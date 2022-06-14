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
import tensorflow as tf
from tensorflow.keras import Model as tf_Model
from bigdl.nano.tf.keras.training_utils import TrainingUtils
from bigdl.nano.tf.keras.inference_utils import InferenceUtils


class Model(TrainingUtils, InferenceUtils):
    """A wrapper class for tf.keras.Model adding more functions for BigDL-Nano."""

    def __init__(self, *args, **kwargs):
        """Create a nano Sequential model, having the same arguments with tf.keras.Sequential."""
        self.model = tf_Model(*args, **kwargs)
        super().__init__()


def f_wapper(f):
    """A wapper function to overide all tf.keras.Model method."""
    def inner(self, *args, **kwargs):
        return f(self.model, *args, **kwargs)
    return inner


for name in dir(tf_Model):
    if name != "fit" and not name.startswith('__'):
        setattr(Model, name, f_wapper(getattr(tf_Model, name)))
