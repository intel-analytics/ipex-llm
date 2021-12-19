
**In this guide we will describe how to scale out TensorFlow (v1.15) programs using Orca in 4 simple steps.**

<a target="_blank" href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/tf_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>&nbsp; <a target="_blank" href="https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/tf_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>

### **Step 0: Prepare Environment**

We recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/python.html) for more details.

**Note:** Conda environment is required to run on the distributed cluster, but not strictly necessary for running on the local machine.

```bash
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo # install either version 0.9 or latest nightly build
pip install tensorflow==1.15.0
pip install tensorflow-datasets==2.0
pip install psutil
```

### **Step 1: Init Orca Context**
```python
if args.cluster_mode == "local":  
    init_orca_context(cluster_mode="local", cores=4)# run in local mode
elif args.cluster_mode == "k8s":  
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2) # run on K8s cluster
elif args.cluster_mode == "yarn":  
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2) # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when you run on Hadoop YARN cluster.

### **Step 2: Define the Model**

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
### **Step 3: Define Train Dataset**

You can define the dataset using standard [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Orca also supports [Spark DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html) and [Orca XShards](./data).

```python
import tensorflow_datasets as tfds

def preprocess(data):
    data['image'] = tf.cast(data["image"], tf.float32) / 255.
    return data['image'], data['label']

# get DataSet
mnist_train = tfds.load(name="mnist", split="train", data_dir=dataset_dir)
mnist_test = tfds.load(name="mnist", split="test", data_dir=dataset_dir)

mnist_train = mnist_train.map(preprocess)
mnist_test = mnist_test.map(preprocess)
```

### **Step 4: Fit with Orca Estimator**

First, create an Estimator.

```python
from zoo.orca.learn.tf.estimator import Estimator

est = Estimator.from_graph(inputs=images,
                           outputs=logits,
                           labels=labels,
                           loss=loss,
                           optimizer=tf.train.AdamOptimizer(),
                           metrics={"acc": acc})
```

Next, fit and evaluate using the Estimator.
```python
est.fit(data=train_dataset,
        batch_size=320,
        epochs=5,
        validation_data=mnist_test)

result = est.evaluate(mnist_test)
print(result)
```

That's it, the same code can run seamlessly in your local laptop and the distribute K8s or Hadoop cluster.

**Note:** You should call `stop_orca_context()` when your program finishes.
