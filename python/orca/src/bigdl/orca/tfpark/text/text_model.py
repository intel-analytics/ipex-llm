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

from zoo.tfpark import KerasModel


# TODO: add word embedding file support
class TextKerasModel(KerasModel):
    """
    The base class for text models in tfpark.
    """
    def __init__(self, labor, optimizer=None, **kwargs):
        self.labor = labor
        self.labor.build(**kwargs)
        model = self.labor.model
        # Recompile the model if user uses a different optimizer other than the default one.
        if optimizer:
            model.compile(loss=model.loss, optimizer=optimizer, metrics=model.metrics)
        super(TextKerasModel, self).__init__(model)

    # Remark: nlp-architect CRF layer has error when directly loaded by tf.keras.models.load_model.
    # Thus we keep the nlp-architect class as labor and uses its save/load,
    # which only saves the weights with model parameters
    # and reconstruct the model using the exact parameters and setting weights when loading.
    def save_model(self, path):
        """
        Save the model to a single HDF5 file.

        :param path: String. The path to save the model.
        """
        self.labor.save(path)

    @staticmethod
    def _load_model(labor, path):
        labor.load(path)
        model = KerasModel(labor.model)
        model.labor = labor
        return model
