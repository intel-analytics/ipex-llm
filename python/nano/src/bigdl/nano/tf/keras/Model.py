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
from time import time
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            perf_tune=False):
        if perf_tune:
            if batch_size is None:
                batch_size = 32
            # Data pipeline
            print("Prefetch Datapipeline")
            assert issubclass(
                type(x), tf.data.Dataset), "Data object currently only support tf.data.dataset"

            batched_data=[]
            batched_target = []
            for i in range(batch_size):
                batched_data_item, batched_target_item = next(iter(x))
                # batched_data.append((batched_data_item, batched_target_item))
                batched_data.append(batched_data_item)
                batched_target.append(batched_target_item)
            start = time()
            super(Model, self).fit(
                batched_data, batched_target, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
                sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq,
                max_queue_size, workers, use_multiprocessing)
            end = time()
            print("Thoughpuy = ", len(batched_data)/(end-start) )

            # 2 x batch_size
            batch_size_2 = int(batch_size * 2)

            print("Use Batch_size: ", batch_size_2)
            start = time()
            super(Model, self).fit(x, y, batch_size_2, epochs, verbose, callbacks, validation_split, validation_data,
                                   shuffle, class_weight,
                                   sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                                   validation_batch_size, validation_freq,
                                   max_queue_size, workers, use_multiprocessing)
            end = time()
            
            print("Thoughpuy = ", len(x)/(end-start) )
            # 1/2 batch_size
            batch_size_12 = int(batch_size / 2)
            print("Use Batch_size: ", batch_size_12)

            super(Model, self).fit(x, y, batch_size_12, epochs, verbose, callbacks, validation_split, validation_data,
                                   shuffle, class_weight,
                                   sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                                   validation_batch_size, validation_freq,
                                   max_queue_size, workers, use_multiprocessing)
            # Without Callbacks
            print("Without Callback")
            if callbacks != None:
                callbacks = None
            super(Model, self).fit(
                x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
                sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq,
                max_queue_size, workers, use_multiprocessing)

        else:
            super(Model, self).fit(
                x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
                sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq,
                max_queue_size, workers, use_multiprocessing)
