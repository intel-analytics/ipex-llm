# Distributed Tensoflow on Spark/BigDL

Analytics-Zoo provides a set APIs to for running tensorflow model in a distributed fashion.

## Concepts
- **TFDatasets** represents a distributed collection of elements to be feed into Tensorflow graph.
TFDatasets can be created using a RDD and each of its records is a list of numpy.ndarray representing
the tensors to be feed into tensorflow graph on each iteration. TFDatasets must be used with the
TFOptimizer or TFPredictor that we will describe next.

- **TFOptimizer** is the class that does all the hard work in distributed training, such as model
distribution and parameter synchronization. It takes the **loss** (a scalar tensor) as input and runs
stochastic gradient descent using the given **optimMethod** on all the **Variables** that contributing
to this loss.

- **TFPredictor** takes a list of tensorflow tensors as the model outputs and feed all the elements in
TFDatasets to produce those outputs and returns a Spark RDD with each of its elements representing the
model prediction for the corresponding input elements.

## Training

1.Data wrangling and analysis using PySpark

```python
from zoo import init_nncontext
from zoo.pipeline.api.net import TFDataset
from tensorflow as tf

sc = init_nncontext()

# Each record in the train_rdd consists of a list of NumPy ndrrays
train_rdd = sc.parallelize(file_list)
  .map(lambda x: read_image_and_label(x))
  .map(lambda image_label: decode_to_ndarrays(image_label))

# TFDataset represents a distributed set of elements,
# in which each element contains one or more Tensorflow Tensor objects. 
dataset = TFDataset.from_rdd(train_rdd,
                             names=["features", "labels"],
                             shapes=[[28, 28, 1], [1]],
                             types=[tf.float32, tf.int32],
                             batch_size=BATCH_SIZE)
```

2.Deep learning model development using TensorFlow

```python
import tensorflow as tf

slim = tf.contrib.slim

images, labels = dataset.tensors
squeezed_labels = tf.squeeze(labels)
with slim.arg_scope(lenet.lenet_arg_scope()):
     logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=squeezed_labels))
```
   
3.Distributed training on Spark and BigDL

```python
from zoo.pipeline.api.net import TFOptimizer
from bigdl.optim.optimizer import MaxIteration, Adam, MaxEpoch, TrainSummary

optimizer = TFOptimizer(loss, Adam(1e-3))
optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

4.Save the variable to checkpoint
   
```python
saver = tf.train.Saver()
saver.save(optimizer.sess, "/tmp/lenet/")
```

### Inference

1.Data wrangling and analysis using PySpark

```python
from zoo import init_nncontext
from zoo.pipeline.api.net import TFDataset
from tensorflow as tf

sc = init_nncontext()

# Each record in the train_rdd consists of a list of NumPy ndrrays
testing_rdd = sc.parallelize(file_list)
  .map(lambda x: read_image_and_label(x))
  .map(lambda image_label: decode_to_ndarrays(image_label))

# TFDataset represents a distributed set of elements,
# in which each element contains one or more Tensorflow Tensor objects. 
dataset = TFDataset.from_rdd(testing_rdd,
                             names=["features"],
                             shapes=[[28, 28, 1]],
                             types=[tf.float32])
```
   
2.Reconstruct the model for inference and load the checkpoint

```python
import tensorflow as tf

slim = tf.contrib.slim

images, labels = dataset.tensors
with slim.arg_scope(lenet.lenet_arg_scope()):
     logits, end_points = lenet.lenet(images, num_classes=10, is_training=False)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "/tmp/lenet")
```

3.Run predictions

```python
predictor = TFPredictor(sess, [logits])
predictions_rdd = predictor.predict()
```

### Relation to TFNet

**TFNet** is a layer representing a tensorflow sub-graph (specified by the inputs and outputs tensor).
It implements the standard BigDL AbstractModule API, it can be used with other Analytics-Zoo/BigDL layers
to construct more complex models for training or inference using the standard Analytics-Zoo/BigDL API. 

You can think of `TFDatasets`, `TFOptimizer`, `TFPredictor` as a set api for training/testing tensorflow models
on Spark/BigDL and resulting a tensorlfow model; while `TFNet` as an Analytics-Zoo layer initialized using tensorflow graph.

For more information on TFNet, please refer to [this](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/APIGuide/PipelineAPI/net.md#tfnet)

