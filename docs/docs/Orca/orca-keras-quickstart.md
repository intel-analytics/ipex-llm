
**In this guide we will describe how to scale out Keras (v2.3) programs using Orca in 4 simple steps.**

<a target="_blank" href="https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>&nbsp; <a target="_blank" href="https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>

### **Step 0: Prepare Environment**

We recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/python.html) for more details.

**Note:** Conda environment is required to run on the distributed cluster, but not strictly necessary for running on the local machine.

```bash
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics_zoo-${VERSION} # install either version 0.9 or latest nightly build
pip install tensorflow==1.15.0
pip install tensorflow-datasets==2.1.0
pip install psutil
pip install pandas
pip install scikit-learn
```

### **Step 1: Init Orca Context**
```python
if args.cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4)# run in local mode
elif args.cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g") # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when you run on Hadoop YARN cluster.

### **Step 2: Define the Model**

You may define your model, loss and metrics in the same way as in any standard (single node) Keras program.

```python
from tensorflow import keras

model = keras.Sequential(
    [keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                         input_shape=(28, 28, 1), padding='valid'),
     keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
     keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                         padding='valid'),
     keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
     keras.layers.Flatten(),
     keras.layers.Dense(500, activation='tanh'),
     keras.layers.Dense(10, activation='softmax'),
     ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
### **Step 3: Define Train Dataset**

You can define the dataset using standard [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Orca also supports [Spark DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html) and [Orca XShards](./data).

```python
import tensorflow as tf
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

est = Estimator.from_keras(keras_model=model)
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

That's it, the same code can run seamlessly in your local laptop and the distribute K8s or Hadoop cluster.

**Note:** You should call `stop_orca_context()` when your program finishes.
