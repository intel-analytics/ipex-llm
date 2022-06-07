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

import sys

from bigdl.dllib.net.utils import find_placeholders, _check_the_same
from bigdl.orca.tfpark.tfnet import TFNet
from bigdl.orca.tfpark.tf_dataset import TFNdarrayDataset, check_data_compatible
from bigdl.orca.tfpark.tf_dataset import _standarize_feature_dataset
from bigdl.dllib.utils.log4Error import invalidInputError

if sys.version >= '3':
    long = int
    unicode = str


class TFPredictor:
    def __init__(self, sess, outputs, inputs=None, dataset=None):
        '''
        TFPredictor takes a list of TensorFlow tensors as the model outputs and
        feed all the elements in TFDatasets to produce those outputs and returns
        a Spark RDD with each of its elements representing the model prediction
        for the corresponding input elements.

        :param sess: the current TensorFlow Session, you should first use this session
        to load the trained variables then pass into TFPredictor
        :param outputs: the output tensors of the TensorFlow model
        '''
        if inputs is None:
            dataset, inputs = TFPredictor._get_datasets_and_inputs(outputs)

        self.sess = sess
        self.dataset = dataset
        self.inputs = inputs
        self.tfnet = TFNet.from_session(sess, self.inputs, outputs)
        if self.dataset.batch_per_thread <= 0:
            invalidInputError(False,
                              "You should set batch_per_thread on TFDataset " +
                              "instead of batch_size for prediction")

    @staticmethod
    def _get_datasets_and_inputs(outputs):
        import tensorflow as tf
        all_required_inputs = find_placeholders(outputs)
        dataset = tf.get_collection(all_required_inputs[0].name)[0]
        inputs = dataset.tensors
        _check_the_same(all_required_inputs, inputs)
        return dataset, inputs

    @classmethod
    def from_outputs(cls, sess, outputs):
        dataset, inputs = TFPredictor._get_datasets_and_inputs(outputs)
        return cls(sess, outputs, inputs, dataset)

    @classmethod
    def from_keras(cls, keras_model, dataset):
        import tensorflow.keras.backend as K
        sess = K.get_session()

        outputs = keras_model.outputs
        inputs = keras_model.inputs

        check_data_compatible(dataset, keras_model, mode="inference")

        if isinstance(dataset, TFNdarrayDataset):
            dataset = _standarize_feature_dataset(dataset, keras_model)

        return cls(sess, outputs, inputs, dataset)

    def predict(self):

        return self.tfnet.predict(self.dataset.get_prediction_data(), mini_batch=True)
