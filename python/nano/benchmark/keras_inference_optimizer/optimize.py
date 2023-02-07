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


import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from bigdl.nano.tf.keras import InferenceOptimizer


def optimize():
    save_dir = "models"

    model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
    input_examples = np.random.random((100, 40, 40, 3))
    input_labels = np.random.randint(0, 10, size=100)

    if "OMP_NUM_THREADS" in os.environ:
        thread_num = int(os.environ["OMP_NUM_THREADS"])
    else:
        thread_num = None

    opt = InferenceOptimizer()
    opt.optimize(
        model=model,
        x=input_examples,
        y=input_labels,
        thread_num=thread_num,
        latency_sample_num=10
    )

    os.makedirs(save_dir, exist_ok=True)
    options = list(InferenceOptimizer.ALL_INFERENCE_ACCELERATION_METHOD.keys())
    for option in options:
        try:
            model = opt.get_model(option)
            opt.save(model, os.path.join(save_dir, option))
        except Exception:
            pass


if __name__ == '__main__':
    optimize()
