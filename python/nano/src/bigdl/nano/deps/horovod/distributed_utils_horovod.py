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
import json
import shutil
from tempfile import TemporaryDirectory
from contextlib import closing
import socket
import tensorflow as tf
from bigdl.nano.utils.common import schedule_processors
from bigdl.nano.utils.common import invalidInputError


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def rebatch_dataset(dataset, num_workers, worker_index):

    from tensorflow.python.data.experimental.ops import distribute
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import math_ops

    batch_size = distribute.compute_batch_size(dataset)

    def apply_rebatch():
        batch_sizes = distribute.batch_sizes_for_worker(
            batch_size, num_workers, 1, worker_index)
        return distribute._RebatchDataset(
            dataset, batch_sizes).prefetch(1)

    def apply_legacy_rebatch():
        return distribute._LegacyRebatchDataset(
            dataset, num_workers).prefetch(1)

    with ops.colocate_with(dataset._variant_tensor):
        return control_flow_ops.cond(
            math_ops.not_equal(batch_size, -1),
            true_fn=apply_rebatch,
            false_fn=apply_legacy_rebatch)


def train_func(envs, model_dir, ds_graph, elem_spec,
               val_ds_graph, val_elem_sepc, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    import horovod.tensorflow.keras as hvd

    os.environ.update(envs[hvd.rank()])

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["OMP_NUM_THREADS"]))

    import horovod.tensorflow.keras as hvd
    new_model = hvd.load_model(os.path.join(model_dir, "temp_model"))
    train_dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)
    if val_ds_graph is not None:
        val_dataset = deserialize_dataset_from_graph(val_ds_graph, val_elem_sepc)
    else:
        val_dataset = None

    from tensorflow.python.distribute.input_ops import auto_shard_dataset
    train_dataset = rebatch_dataset(train_dataset, hvd.size(), hvd.rank())
    train_dataset = auto_shard_dataset(train_dataset, hvd.size(), hvd.rank(), hvd.size())

    if val_dataset is not None:
        val_dataset = rebatch_dataset(val_dataset, hvd.size(), hvd.rank())
        val_dataset = auto_shard_dataset(val_dataset, hvd.size(), hvd.rank(), hvd.size())

    task_id = hvd.rank()

    if task_id == 0:
        verbose = fit_kwargs['verbose']
    else:
        verbose = 0
    del fit_kwargs['verbose']
    history = new_model.fit(train_dataset,
                            validation_data=val_dataset,
                            verbose=verbose,
                            **fit_kwargs)
    if task_id == 0:
        path = os.path.join(model_dir, 'trained_model_weights')
        new_model.save_weights(path, overwrite=True)
    else:
        path = os.path.join(model_dir, f'trained_model_weights_{task_id}')
        new_model.save_weights(path, overwrite=True)
    return history


def distributed_train_keras(backend, model, nprocs, fit_kwargs=None):

    if fit_kwargs is None:
        fit_kwargs = {}

    envs = schedule_processors(nprocs)

    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph

    train_dataset = fit_kwargs.pop('x')
    val_dataset = fit_kwargs.pop('validation_data')

    train_ds_def = serialize_dataset_to_graph(train_dataset).numpy()
    train_elem_spec = train_dataset.element_spec

    if val_dataset is not None:
        val_ds_def = serialize_dataset_to_graph(val_dataset).numpy()
        val_elem_spec = val_dataset.element_spec
    else:
        val_ds_def = None
        val_elem_spec = None

    # this is to work around a tensorflow bug: https://github.com/keras-team/keras/issues/16023
    model.evaluate(train_dataset, verbose=0, steps=1)
    invalidInputError(model.compiled_metrics.built, "model.compiled_metrics.built should be True")
    with TemporaryDirectory() as temp_dir:
        model.save(os.path.join(temp_dir, 'temp_model'))

        train_args = (temp_dir, train_ds_def, train_elem_spec,
                      val_ds_def, val_elem_spec, fit_kwargs)

        histrories = backend.run(target=train_func,
                                 args=train_args,
                                 nprocs=nprocs,
                                 envs=envs)
        model.load_weights(os.path.join(temp_dir, 'trained_model_weights'))
    return histrories[0]
