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
from bigdl.nano.tf.keras.inference_utils import InferenceUtils
from bigdl.nano.tf.keras.training_utils import TrainingUtils


class Sequential(TrainingUtils, InferenceUtils, tf.keras.Sequential):
    """A wrapper class for tf.keras.Sequential adding more functions for BigDL-Nano."""

    def __init__(self, *args, **kwargs):
        """Create a nano Sequential model, having the same arguments with tf.keras.Sequential."""
        super().__init__(*args, **kwargs)
