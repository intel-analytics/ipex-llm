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

from bigdl.nano.deps.onnxruntime.tensorflow.tensorflow_onnxruntime_model \
    import KerasONNXRuntimeModel

from ..quantization import BaseONNXRuntimeQuantization
from ...tensorflow import TensorflowQuantization
from .metric import KerasONNXRuntimeINCMetic


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
            calib_dataset = KerasNumpyDataset(calib_dataset, model.dtype)

        if isinstance(model, KerasONNXRuntimeModel):
            model = model.onnx_model

        return model, calib_dataset, metric

    def _post_execution(self, q_model):
        return KerasONNXRuntimeModel(q_model.model, input_sample=None,
                                     onnxruntime_session_options=self.session_options)


# we cannot call `.numpy()` in `Dataset.map()`,
# so we use this wrapper to transform the output of dataset to numpy array
class KerasNumpyDataset():
    def __init__(self, dataset, dtype):
        self.dataset = dataset
        self.batch_size = 1     # it's necessary
        self.dtype = dtype      # the dtype of dataset and model must be exactly the same

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for batch in self.dataset:
            yield tuple(
                map(lambda x: x.numpy().astype(self.dtype) if isinstance(x, tf.Tensor) else x,
                    batch)
            )
