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
from tensorflow.keras import Model as Model
from bigdl.nano.tf.keras.training_utils import TrainingUtils
from bigdl.nano.tf.keras import InferenceOptimizer
from bigdl.nano.tf.keras.inheritance_utils import inject_function
from keras.engine.training import Model as V2Model
from keras.engine.training_v1 import Model as V1Model


# override_method(Model, TFModel, f_wrapper)
V1Model.__doc__ = 'A wrapper class for tf.keras.Model adding more functions for BigDL-Nano.'
V2Model.__doc__ = 'A wrapper class for tf.keras.Model adding more functions for BigDL-Nano.'
inject_function(V2Model, TrainingUtils, InferenceOptimizer)
inject_function(V1Model, TrainingUtils, InferenceOptimizer)
