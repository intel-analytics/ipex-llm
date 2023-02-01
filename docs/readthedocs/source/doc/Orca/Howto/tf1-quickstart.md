# Scale TensorFlow 1.15 Applications

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/main/python/orca/colab-notebook/quickstart/tf_lenet_mnist.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/main/python/orca/colab-notebook/quickstart/tf_lenet_mnist.ipynb)

---

**In this guide we will describe how to scale out _TensorFlow 1.15_ programs using Orca in 4 simple steps.**

### Step 0: Prepare Environment

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../Overview/install.md) for more details.

```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37

pip install bigdl-orca
pip install tensorflow==1.15
pip install tensorflow-datasets==2.0
pip install psutil
```

### Step 1: Init Orca Context
```python
from bigdl.orca import init_orca_context, stop_orca_context

if cluster_mode == "local":  # For local machine
    init_orca_context(cluster_mode="local", cores=4, memory="4g")
elif cluster_mode == "k8s":  # For K8s cluster
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, memory="4g", master=..., container_image=...)
elif cluster_mode == "yarn":  # For Hadoop/YARN cluster
    init_orca_context(cluster_mode="yarn", num_nodes=2, cores=2, memory="4g")
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](../Overview/orca-context.md) for more details.

Please check the tutorials if you want to run on [Kubernetes](../Tutorial/k8s.md) or [Hadoop/YARN](../Tutorial/yarn.md) clusters.

### Step 2: Define the Model

You may define your model, loss and metrics in the same way as in any standard (single node) TensorFlow program.

```python
import tensorflow as tf

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
### Step 3: Define Train Dataset

You can define the dataset using standard [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Orca also supports [Spark DataFrame](./spark-dataframe.md) and [Orca XShards](./xshards-pandas.md).

```python
import tensorflow_datasets as tfds

def preprocess(data):
    data['image'] = tf.cast(data["image"], tf.float32) / 255.
    return data['image'], data['label']

# get DataSet
mnist_train = tfds.load(name="mnist", split="train", data_dir=...)
mnist_test = tfds.load(name="mnist", split="test", data_dir=...)

mnist_train = mnist_train.map(preprocess)
mnist_test = mnist_test.map(preprocess)
```

### Step 4: Fit with Orca Estimator

First, create an Orca Estimator for TensorFlow.

```python
from bigdl.orca.learn.tf.estimator import Estimator

est = Estimator.from_graph(inputs=images,
                           outputs=logits,
                           labels=labels,
                           loss=loss,
                           optimizer=tf.train.AdamOptimizer(),
                           metrics={"acc": acc})
```

Next, fit and evaluate using the Estimator.
```python
est.fit(data=mnist_train,
        batch_size=320,
        epochs=5,
        validation_data=mnist_test)

result = est.evaluate(mnist_test)
print(result)
```

**Note:** You should call `stop_orca_context()` when your program finishes.

That's it, the same code can run seamlessly on your local laptop and scale to [Kubernetes](../Tutorial/k8s.md) or [Hadoop/YARN](../Tutorial/yarn.md) clusters.
