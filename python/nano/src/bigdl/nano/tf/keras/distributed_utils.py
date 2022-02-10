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

from turtle import back
from bigdl.nano.common.cpu_schedule import schedule_workers
import os
import json
import shutil
from tempfile import TemporaryDirectory
from contextlib import closing
import socket
import tensorflow as tf

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def train_func(model_dir, ds_graph, elem_spec, epochs, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        new_model = tf.keras.models.load_model('/tmp/temp_model')
        dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)
        tf_config = json.loads(os.environ["TF_CONFIG"])
        if tf_config["task"]["index"] == 0:
            verbose = 1
        else:
            verbose = 0
        new_model.fit(dataset, epochs=epochs, verbose=verbose, **fit_kwargs)
        task_id = tf_config["task"]["index"]
        if task_id == 0:
            print("saving model")
            new_model.save(os.path.join(model_dir, 'temp_model'), overwrite=True)
        else:
            new_model.save(os.path.join(model_dir, f'temp_model_{task_id}'), overwrite=True)
            shutil.rmtree(os.path.join(model_dir, f'temp_model_{task_id}'))

def distributed_train_keras(backend, model, train_dataset, nprocs, epochs, fit_kwargs=None):

    backend.setup()

    if fit_kwargs is None:
        fit_kwargs = {}

    cpu_procs = schedule_workers(nprocs)

    from tensorflow.python.distribute.input_lib import _dummy_tensor_fn
    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph

    graph_def = serialize_dataset_to_graph(train_dataset)
    graph_def = graph_def.numpy()
    elem_spec = train_dataset.element_spec

    # this is to work around a tensorflow problem: if we save before calling fit, the saved format is incorrect
    # dummy_batch is a batch of input with batch size equal to 0, so that the model.fit does not take any effect
    dummy_batch = _dummy_tensor_fn(elem_spec)
    model.fit(tf.data.Dataset.from_tensors(dummy_batch))
    
    ports = set()
    while len(ports) < nprocs:
        ports.add(find_free_port())
    ports = list(ports)
    worker_list = [f"localhost:{p}" for p in ports]

    with TemporaryDirectory() as temp_dir:
        model.save(os.path.join(temp_dir, 'temp_model'))

        envs = []
        for i in range(nprocs):
            env = {
                "KMP_AFFINITY": f"granularity=fine,proclist"\
                                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                "TF_CONFIG": json.dumps({
                    'cluster': {
                        'worker': worker_list
                        },
                    'task': {'type': 'worker', 'index': i}
                    }),
                'no_proxy': "localhost",
            }
            envs.append(env)
        
        backend.run(target=train_func,
                    args=(temp_dir, graph_def, elem_spec, epochs, fit_kwargs),
                    nprocs=nprocs,
                    envs=envs)
        model = tf.keras.models.load_model(os.path.join(temp_dir, 'temp_model'))
    return model


        


