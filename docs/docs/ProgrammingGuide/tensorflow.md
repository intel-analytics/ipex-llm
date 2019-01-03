Analytics-Zoo provides a set APIs for running TensorFlow model on Spark in a distributed fashion.

## System Requirement
TensorFlow version: 1.10

OS version (all 64-bit): __Ubuntu 16.04 or later__, __macOS 10.12.6 or later__, __Windows 7 or later__ (TensorFlow is
 only tested and supported on these 64-bit systems as stated [here](https://www.tensorflow.org/install/)).
 
To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


## Concepts
- **TFDatasets** represents a distributed collection of elements to be fed into a TensorFlow graph.
TFDatasets can be created directly from an RDD; each record in the RDD should be a list of numpy.ndarray
representing the input data. TFDatasets must be used with the TFOptimizer or TFPredictor (to be described next).

- **TFOptimizer** is the class that does all the hard work in distributed training, such as model
distribution and parameter synchronization. It takes the user specified **loss** (a TensorFlow scalar tensor) as
an argument and runs stochastic gradient descent using the given **optimMethod** on all the **Variables** that
contribute to this loss.

- **TFPredictor** takes a list of user specified TensorFlow tensors as the model outputs, and feed all the
elements in TFDatasets to produce those outputs; it returns a Spark RDD with each of its records representing the
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
# in which each element contains one or more TensorFlow Tensor objects. 
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

You can also construct your model using Keras provided by Tensorflow.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

data = Input(shape=[28, 28, 1])

x = Flatten()(data)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=data, outputs=predictions)

model.compile(optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
```
   
3.Distributed training on Spark and BigDL

```python
from zoo.pipeline.api.net import TFOptimizer
from bigdl.optim.optimizer import MaxIteration, Adam, MaxEpoch, TrainSummary

optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

For Keras model:

```python
from zoo.pipeline.api.net import TFOptimizer
from bigdl.optim.optimizer import MaxIteration, MaxEpoch, TrainSummary

optimizer = TFOptimizer.from_keras(keras_model=model, dataset=dataset)
optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

4.Save the variable to checkpoint
   
```python
saver = tf.train.Saver()
saver.save(optimizer.sess, "/tmp/lenet/")
```

For Keras model, you can also Keras' `save_weights` api.

```python
model.save_weights("/tmp/keras.h5")
```

### Inference

1.Data processing using PySpark

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
# in which each element contains one or more TensorFlow Tensor objects. 
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

As before, you can also construct and restore your model using Keras provided by Tensorflow.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

data = Input(shape=[28, 28, 1])

x = Flatten()(data)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=data, outputs=predictions)

model.load_weights("/tmp/mnist_keras.h5")

```

3.Run predictions

```python
predictor = TFPredictor.from_outputs(sess, [logits])
predictions_rdd = predictor.predict()
```

For keras model:

```python
predictor = TFPredictor.from_keras(model, dataset)
predictions_rdd = predictor.predict()
```

### Relationship to TFNet

**TFNet** is a layer representing a TensorFlow sub-graph (specified by the input and output TensorFlow tensors).
It implements the standard BigDL layer API, and can be used with other Analytics-Zoo/BigDL layers
to construct more complex models for training or inference using the standard Analytics-Zoo/BigDL API. 

You can think of `TFDatasets`, `TFOptimizer`, `TFPredictor` as a set API for training/testing TensorFlow models
on Spark/BigDL, while `TFNet` as an Analytics-Zoo/BigDL layer initialized using TensorFlow graph.

For more information on TFNet, please refer to the [API Guide](../APIGuide/PipelineAPI/net.md#tfnet)

