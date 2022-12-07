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


# Required Dependecies
#
# ```bash
# pip install tf2onnx onnx onnxruntime
# ```


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50

from bigdl.nano.tf.keras import Model


def create_dataset(img_size, batch_size):
    dataset, info = tfds.load('imagenette/320px-v2',
                             data_dir='/tmp/data',
                             split='validation[:5%]',
                             with_info=True,
                             as_supervised=True)
    
    num_classes = info.features['label'].num_classes
    def preprocessing(img, label):
        return tf.image.resize(img, (img_size, img_size)), \
            tf.one_hot(label, num_classes)

    dataset = dataset.map(preprocessing).batch(batch_size)
    return dataset


if __name__ == '__main__':
    img_size = 224
    batch_size = 32

    dataset = create_dataset(img_size, batch_size)
    model = ResNet50(weights='imagenet', input_shape=(img_size, img_size, 3))
    preds = model.predict(dataset)

    # Accelerate inference with ONNX
    #
    # Use `Model` or `Sequential` in `bigdl.nano.tf.keras` to create model,
    # then call its `trace` method with `accelerator='onnxruntime'` and
    # pass a `TensorSpec` defining the shape of input to `input_spec` parameter,
    # then you can use the returned model for inference, all inference will be 
    # accelerated automatically after that.
    #
    model = Model(inputs=model.inputs, outputs=model.outputs)
    spec = tf.TensorSpec((None, 224, 224, 3), tf.float32)
    onnx_model = model.trace(accelerator='onnxruntime', input_spec=spec)
    onnx_preds = onnx_model.predict(dataset)

    # Using ONNX to accelerate inference will cause a very tiny loss of precision,
    # which can be completely ignored. Here we check if the relative errors
    # of output are all under 1e-4.
    np.testing.assert_allclose(preds, onnx_preds, rtol=1e-4)
