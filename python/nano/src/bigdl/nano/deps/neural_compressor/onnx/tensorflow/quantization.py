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

import numpy as np
import tensorflow as tf

from bigdl.nano.deps.onnxruntime.tensorflow.tensorflow_onnxruntime_model \
    import KerasONNXRuntimeModel

from ..quantization import BaseONNXRuntimeQuantization
from .metric import KerasONNXRuntimeINCMetic
from bigdl.nano.tf.model import AcceleratedKerasModel


class KerasONNXRuntimeQuantization(BaseONNXRuntimeQuantization):
    def __init__(self, framework='onnxrt_qlinear', **kwargs):
        """
        Create a Intel Neural Compressor Quantization object for ONNXRuntime in Tensorflow.
        """
        kwargs['framework'] = framework
        self.session_options = kwargs.pop('onnxruntime_session_options', None)
        super().__init__(**kwargs)
        self._inc_metric_cls = KerasONNXRuntimeINCMetic

    def _pre_execution(self, model, calib_dataset=None, metric=None):
        if calib_dataset:
            x, y = calib_dataset
            calib_dataset = KerasNumpyDataset(x, y, model.dtype)

        if isinstance(model, KerasONNXRuntimeModel):
            model = model.onnx_model

        return model, calib_dataset, metric

    def _post_execution(self, q_model):
        return KerasONNXRuntimeModel(q_model.model,
                                     onnxruntime_session_options=self.session_options)


# we cannot call `.numpy()` in `Dataset.map()`,
# so we use this wrapper to transform the output of dataset to numpy array
class KerasNumpyDataset():
    def __init__(self, x, y, dtype=tf.float32):
        self.x = x
        self.y = y
        self.batch_size = 1     # it's necessary
        self.dtype = dtype      # the dtype of dataset and model must be exactly the same

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        if isinstance(self.x, tf.data.Dataset):
            for batch in self.x.batch(1):
                yield AcceleratedKerasModel.tensors_to_numpy(batch, self.dtype)
        else:
            for x, y in zip(self.x, self.y):
                x, y = AcceleratedKerasModel.tensors_to_numpy((x, y), self.dtype)
                yield np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
