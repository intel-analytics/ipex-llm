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


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _train_func(model_dir, ds_graph, elem_spec,
                val_ds_graph, val_elem_sepc, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # We must set `auto_shard_policy` to `DATA` or `OFF` when
    # the number of dataset files is less than the number of processes
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    with strategy.scope():
        new_model = tf.keras.models.load_model(os.path.join(model_dir, "temp_model"))
        train_dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)
        train_dataset = train_dataset.with_options(options)
        if val_ds_graph is not None:
            val_dataset = deserialize_dataset_from_graph(val_ds_graph, val_elem_sepc)
            val_dataset = val_dataset.with_options(options)
        else:
            val_dataset = None

        task_id = strategy.cluster_resolver.task_id

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
    """Run pseudo-distributed training on the keras model with the specified backend."""
    backend.setup()

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

    ports = set()
    while len(ports) < nprocs:
        ports.add(_find_free_port())
    ports = list(ports)
    worker_list = [f"localhost:{p}" for p in ports]

    with TemporaryDirectory() as temp_dir:
        model.save(os.path.join(temp_dir, 'temp_model'))

        for i, env in enumerate(envs):
            env.update({
                "TF_CONFIG": json.dumps(
                    {
                        'cluster': {
                            'worker': worker_list
                        },
                        'task': {'type': 'worker', 'index': i}
                    }),
                'no_proxy': "localhost"
            })

        train_args = (temp_dir, train_ds_def, train_elem_spec,
                      val_ds_def, val_elem_spec, fit_kwargs)

        histrories = backend.run(target=_train_func,
                                 args=train_args,
                                 nprocs=nprocs,
                                 envs=envs)
        model.load_weights(os.path.join(temp_dir, 'trained_model_weights'))
    return histrories[0]
