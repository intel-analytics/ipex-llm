import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import ray
# from bigdl.nano.common.cpu_schedule import schedule_workers
from ray.util.queue import Queue
import os
import json
import shutil


@ray.remote
def train_func(ds_graph, elem_spec, epochs, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph, deserialize_dataset_from_graph

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
            new_model.save("/tmp/temp_model", overwrite=True)
        else:
            new_model.save(f"/tmp/temp_model_{task_id}", overwrite=True)
            shutil.rmtree(f"/tmp/temp_model_{task_id}")
        print("fit returning")
        return 0

def distributed_train_keras(model, train_dataset, num_instances, epochs, fit_kwargs=None):
    import ray
    ray.init()
    if fit_kwargs is None:
        fit_kwargs = {}

    # cpu_procs = schedule_workers(nprocs)
    cpu_procs =[[0, 1], [2, 3]]
    nprocs = len(cpu_procs)

    from tensorflow.python.distribute.input_lib import _dummy_tensor_fn
    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph, deserialize_dataset_from_graph

    graph_def = serialize_dataset_to_graph(train_dataset)
    graph_def = graph_def.numpy()
    elem_spec = train_dataset.element_spec
    dummy_batch = _dummy_tensor_fn(elem_spec)
    model.fit(tf.data.Dataset.from_tensors(dummy_batch))
    
    model.save('/tmp/temp_model')

    results = []
    for i in range(nprocs):
        env = {
            "KMP_AFFINITY": f"granularity=fine,proclist"\
                              f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
            "OMP_NUM_THREADS": str(len(cpu_procs[i])),
            "TF_CONFIG": json.dumps({
                'cluster': {
                    'worker': ['localhost:12341', 'localhost:12342']
                    },
                'task': {'type': 'worker', 'index': i}
                }),
            'no_proxy': "localhost",
        }
        runtime_env = {
            "env_vars": env
        }
        
        results.append(train_func.options(runtime_env=runtime_env).remote(graph_def, elem_spec, epochs, fit_kwargs))
    
    ray.get(results)
    ray.shutdown()

    return tf.keras.models.load_model('/tmp/temp_model')

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

distributed_train_keras(model, train_ds, 2, 3)


