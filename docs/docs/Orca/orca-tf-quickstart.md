## **Orca TensorFlow Quickstart**

**In this guide we’ll show you how to organize your TensorFlow code into Orca in 3 steps**

Organizing your code with Orca makes your code:
* Keep all the flexibility
* Easier to reproduce
* Utilize distributed training without changing your model

### **Step 0: Prepare environment**
Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -y -n analytics-zoo python==3.7.7
conda activate analytics-zoo
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl
pip install tensorflow==1.15.0
pip install psutil
```

Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

### **Step 1: Init Orca Context**
```python
import tensorflow as tf
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator

# run in local mode
init_orca_context(cluster_mode="local", cores=4)

# run in yarn client mode
init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g")
```
* Reference: [Orca Context](https://analytics-zoo.github.io/master/#Orca/context/)

### **Step 2: Define Model, Loss Function and Metrics**

#### **For Keras Users**
```python
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
```

#### **For Graph Users**
```python
def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)

def lenet(images):
    with tf.variable_scope('LeNet', [images]):
        net = tf.layers.conv2d(images, 32, (5, 5), activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool1')
        net = tf.layers.conv2d(net, 64, (5, 5), activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool2')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
        logits = tf.layers.dense(net, 10)
        return logits

# tensorflow inputs
images = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
# tensorflow labels
labels = tf.placeholder(dtype=tf.int32, shape=(None,))

logits = lenet(images)

loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

acc = accuracy(logits, labels)
```

### **Step 3: Fit with Orca TensorFlow Estimator**
1. Define the dataset in whatever way you want. Orca just needs tf.data.Dataset, Spark DataFrame or Orca SparkXShards.
```python
def preprocess(x, y):
    return tf.to_float(tf.reshape(x, (-1, 28, 28, 1))) / 255.0, y

# get DataSet
(train_feature, train_label), (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

# tf.data.Dataset.from_tensor_slices is for demo only. For production use, please use
# file-based approach (e.g. tfrecord).
train_dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
train_dataset = train_dataset.map(preprocess)
val_dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
val_dataset = val_dataset.map(preprocess)
```

2. Create an estimator
* For Keras Users
```python
est = Estimator.from_keras(keras_model=model)
```
* For Graph Users
```python
est = Estimator.from_graph(inputs=images, 
                           outputs=logits,
                           labels=labels,
                           loss=loss,
                           optimizer=tf.train.AdamOptimizer(),
                           metrics={"acc": acc})
```

3. Fit with estimator
```python
est.fit(data=train_dataset,
        batch_size=320,
        epochs=max_epoch,
        validation_data=val_dataset)
```

4. Evaluate with estimator
```python
result = est.evaluate(val_dataset)
print(result)
```

5. Save Model
* For Keras Users
```python
est.save_keras_model("/tmp/mnist_keras.h5")
```
* For Graph Users
```python
est.save_tf_checkpoint("/tmp/lenet/model")
```

**Note:** you should call `stop_orca_context()` when your application finishes.