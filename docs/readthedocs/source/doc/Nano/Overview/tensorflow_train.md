# TensorFlow Training

BigDL-Nano can be used to accelerate TensorFlow Keras applications on training workloads. The optimizations in BigDL-Nano are delivered through

- BigDL-Nano's `Model` and `Sequential` classes, which have identical APIs with `tf.keras.Model` and `tf.keras.Sequential` with an enhanced `fit` method.
- BigDL-Nano's decorator `nano` (potentially with the help of `nano_multiprocessing` and `nano_multiprocessing_loss`) to handle keras model with customized training loop.

We will briefly describe here the major features in BigDL-Nano for TensorFlow training.

### Best Known Configurations
When you install BigDL-Nano by `pip install bigdl-nano[tensorflow]`, `intel-tensorflow` will be installed in your environment, which has intel's oneDNN optimizations enabled by default; and when you run `source bigdl-nano-init`, it will export a few environment variables, such as `OMP_NUM_THREADS` and `KMP_AFFINITY`, according to your current hardware. Empirically, these environment variables work best for most TensorFlow applications. After setting these environment variables, you can just run your applications as usual (`python app.py`) and no additional changes are required.

### Multi-Instance Training
When training on a server with dozens of CPU cores, it is often beneficial to use multiple training instances in a data-parallel fashion to make full use of the CPU cores. However

- Naively using TensorFlow's `MultiWorkerMirroredStrategy` can cause conflict in CPU cores and often cannot provide performance benefits.
- Customized training loop could be hard to use together with `MultiWorkerMirroredStrategy`

BigDL-Nano makes it very easy to conduct multi-instance training correctly for default/customized training loop models.

#### Keras Model with default training loop
 You can just set the `num_processes` parameter in the `fit` method in your `Model` or `Sequential` object and BigDL-Nano will launch the specific number of processes to perform data-parallel training. Each process will be automatically pinned to a different subset of CPU cores to avoid conflict and maximize training throughput.

```python
import tensorflow as tf
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
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

model.fit(train_ds, epochs=3, validation_data=val_ds, num_processes=2)
```

#### Keras Model with customized training loop

To make them run in a multi-process way, you may only add 2 lines of code.

- add `nano_multiprocessing` to the `train_step` function with gradient calculation and applying process.
- add `@nano(num_processes=...)` to the training loop function with iteration over full dataset.

```python
from bigdl.nano.tf.keras import nano_multiprocessing, nano
import tensorflow as tf

tf.random.set_seed(0)
global_batch_size = 32

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
optimizer = tf.keras.optimizers.SGD()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(128).batch(
    global_batch_size)

@nano_multiprocessing  # <-- Just remove this line to run on 1 process
@tf.function
def train_step(inputs, model, loss_object, optimizer):
    features, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@nano(num_processes=4)  # <-- Just remove this line to run on 1 process
def train_whole_data(model, dataset, loss_object, optimizer, train_step):
    for inputs in dataset:
        print(train_step(inputs, model, loss_object, optimizer))
```


Note that, different from the conventions in [BigDL-Nano PyTorch multi-instance training](./pytorch_train.html#multi-instance-training), the effective batch size will not change in TensorFlow multi-instance training, which means it is still the batch size you specify in your dataset. This is because TensorFlow's `MultiWorkerMirroredStrategy` will try to split the batch into multiple sub-batches for different workers. We chose this behavior to match the semantics of TensorFlow distributed training.

When you do want to increase your effective `batch_size`, you can do so by directly changing it in your dataset definition and you may also want to gradually increase the learning rate linearly to the `batch_size`, as described in this [paper](https://arxiv.org/abs/1706.02677) published by Facebook.