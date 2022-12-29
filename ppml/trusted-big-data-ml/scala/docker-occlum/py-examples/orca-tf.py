from bigdl.orca import init_orca_context, stop_orca_context

cluster_mode = "local"
if cluster_mode == "local":  # For local machine
    init_orca_context(cluster_mode="spark-submit")
elif cluster_mode == "k8s":  # For K8s cluster
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1)
elif cluster_mode == "yarn":  # For Hadoop/YARN cluster
    init_orca_context(cluster_mode="yarn", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1)

import tensorflow as tf

def model_creator(config):
    model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                input_shape=(28, 28, 1), padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(500, activation='tanh'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def preprocess(x, y):
    x = tf.cast(tf.reshape(x, (28, 28, 1)), dtype=tf.float32) / 255.0
    return x, y

def train_data_creator(config, batch_size):
    (train_feature, train_label), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset

def val_data_creator(config, batch_size):
    _, (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    return dataset

from bigdl.orca.learn.tf2 import Estimator

est = Estimator.from_keras(log_to_driver=False, backend="spark", model_creator=model_creator, workers_per_node=1)

batch_size = 320
stats = est.fit(train_data_creator,
                epochs=3,
                batch_size=batch_size,
                steps_per_epoch=60000 // batch_size,
                validation_data=val_data_creator,
                validation_steps=10000 // batch_size)

est.save("/tmp/mnist_keras.ckpt")

stats = est.evaluate(val_data_creator, num_steps=10000 // batch_size)
est.shutdown()
print(stats)


stop_orca_context()