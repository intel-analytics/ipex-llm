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


class TrainingUtils:
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
            nprocs=None,
            backend="multiprocessing"):

        fit_kwargs = dict(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        if nprocs is not None:
            if validation_data is not None:
                msg = "validataion_data must be a tf.data.Dataset for multi-process training"
                assert isinstance(x, (tf.compat.v1.data.Dataset, tf.data.Dataset)), msg
            msg = "x must be a tf.data.Dataset for multi-process training"
            assert isinstance(x, (tf.compat.v1.data.Dataset, tf.data.Dataset)), msg

            if backend == "horovod":
                from bigdl.nano.common.multiprocessing.multiprocs_backend import HorovodBackend
                _backend = HorovodBackend()
                from bigdl.nano.tf.keras.distributed_utils_horovod import distributed_train_keras
                history = distributed_train_keras(_backend,
                                                  model=self,
                                                  nprocs=nprocs,
                                                  fit_kwargs=fit_kwargs)
                return history

            else:

                if backend == "multiprocessing":
                    from bigdl.nano.common.multiprocessing.multiprocs_backend \
                        import MultiprocessingBackend
                    _backend = MultiprocessingBackend()
                elif backend == "ray":
                    from bigdl.nano.common.multiprocessing.ray_backend import RayBackend
                    _backend = RayBackend()
                else:
                    raise NotImplementedError("Backend {} is not implemented.".format(backend))
                from bigdl.nano.tf.keras.distributed_utils import distributed_train_keras
                history = distributed_train_keras(_backend,
                                                  model=self,
                                                  nprocs=nprocs,
                                                  fit_kwargs=fit_kwargs)
                return history
        else:
            return super().fit(**fit_kwargs)
