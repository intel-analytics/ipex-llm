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


import time
import os
import sys
import argparse

import numpy as np
from tensorflow.keras.applications import ResNet50
from bigdl.nano.tf.keras import InferenceOptimizer


def run(args):
    save_dir = "models"
    imgs = np.random.random((100, 40, 40, 3))
    opt = InferenceOptimizer()
    options = list(InferenceOptimizer.ALL_INFERENCE_ACCELERATION_METHOD.keys())
    if args.option in options:
        try:
            try:
                model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
                model = opt.load(os.path.join(save_dir, args.option), model)
            except Exception:
                model = opt.load(os.path.join(save_dir, args.option))
            # warmup
            model.predict(imgs[:10], batch_size=1)

            st = time.time()
            model.predict(imgs, batch_size=1)
            end = time.time()
            throughput = len(imgs) / (end - st)
        except Exception:
            throughput = 0
        print(f"Throughput: {throughput}")
    else:
        print(f"unkonwn option: {args.option}")
        sys.exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str)
    args = parser.parse_args()

    run(args)
