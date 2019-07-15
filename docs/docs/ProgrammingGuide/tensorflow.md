Analytics-Zoo provides a set APIs for running TensorFlow model on Spark in a distributed fashion.

# System Requirement
TensorFlow version: 1.10

OS version (all 64-bit): __Ubuntu 16.04 or later__, __macOS 10.12.6 or later__, __Windows 7 or later__ (TensorFlow is
 only tested and supported on these 64-bit systems as stated [here](https://www.tensorflow.org/install/)).
 
To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

# TFPark API

TFPark is a set of high-level api modeling after tf.keras and tf.estimator to help user to train and evaluate TensorFlow
models on Spark and BigDL. Users can define their model using `tf.keras` API or using `model_fn` similar to `tf.estimator`.

## TFDataset

**TFDatasets** represents a distributed collection of elements (backed by a RDD) to be fed into a TensorFlow graph.
TFDatasets can be created from numpy.ndarrays, an rdd of numpy.ndarrays as well as ImageSet, TextSet and FeatureSet.
It acts as an interface connecting RDD data to TensorFlow models.

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
                             features=(tf.float32, [28, 28, 1]),
                             labels=(tf.int32, []),
                             batch_size=BATCH_SIZE)
```

More on TFDataset API [API Guide](../APIGuide/PipelineAPI/net.md#tfnet)

## KerasModel

KerasModel enables user to use `tf.keras` API to define TensorFlow models and perform training or evaluation on top
of Spark and BigDL in a distributed fashion.

1. Create a KerasModel
```python
from zoo.tfpark import KerasModel, TFDataset
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax'),
     ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
keras_model = KerasModel(model)

```

2. Perform training on TFDataset and save model

```python
keras_model.fit(training_dataset, epochs=max_epoch)

model.save_weights("/tmp/model.h5")

```

2. Loading saved model and preform evaluation or inference

```python
model.load_weights("/tmp/model.h5")

evaluation_results = model.evaluate(eval_dataset)

predictions = model.predict(pred_dataset)
```

More on KerasModel API [API Guide](../APIGuide/TFPark/model.md)

## TFEstimator

TFEstimator wraps a model defined by `model_fn`. The `model_fn` is almost identical to TensorFlow's `model_fn`
except users are required to return a `TFEstimator` object. Users do not need to construct backward graph
(calling `optimizer.minimize(...)`) but set a `loss` tensor in `TFEstimator`.

1. Define a `model_fn`

```python
import tensorflow as tf
from zoo.tfpark.estimator import TFEstimator, TFEstimatorSpec
def model_fn(features, labels, mode):

    hidden = tf.layers.dense(features, 32, activation=tf.nn.relu)
    
    logits = tf.layers.dense(hidden, 10)

    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
        return TFEstimatorSpec(mode, predictions=logits, loss=loss)
    else:
        return TFEstimatorSpec(mode, predictions=logits)

```

2. Define a input_fn

```python
import tensorflow as tf
sc = init_nncontext()
def input_fn(mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        training_rdd = get_data_rdd("train", sc)
        dataset = TFDataset.from_rdd(training_rdd,
                                     features=(tf.float32, [28, 28, 1]),
                                     labels=(tf.int32, []),
                                     batch_size=320)
    elif mode == tf.estimator.ModeKeys.EVAL:
        validation_rdd = get_data_rdd("validation", sc)
        dataset = TFDataset.from_rdd(testing_rdd,
                                     features=(tf.float32, [28, 28, 1]),
                                     labels=(tf.int32, []),
                                     batch_size=320)
    else:
        testing_rdd = get_data_rdd("test", sc)
        dataset = TFDataset.from_rdd(testing_rdd,
                                     features=(tf.float32, [28, 28, 1]),
                                     batch_per_thread=80)
    return dataset
```

3. Create TFEstimator and perform training, evaluation or inference

```python
estimator = TFEstimator(model_fn, tf.train.AdamOptimizer(), model_dir="/tmp/estimator")
estimator.train(input_fn, steps=10000)
evaluation_result = estimator.evaluate(input_fn, ["acc"])
predictions = estimator.predict(input_fn)
```

More on TFEstimator API [API Guide](../APIGuide/TFPark/estimator.md)

# Low level API

## Concepts

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
                             features=(tf.float32, [28, 28, 1]),
                             labels=(tf.int32, []),
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
from zoo.tfpark import TFOptimizer
from bigdl.optim.optimizer import MaxIteration, Adam, MaxEpoch, TrainSummary

optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
optimizer.set_train_summary(TrainSummary("/tmp/az_lenet", "lenet"))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

For Keras model:

```python
from zoo.tfpark import TFOptimizer
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
                             features=(tf.float32, [28, 28, 1]),
                             batch_per_thread=4)
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

# Relationship to TFNet

**TFNet** is a layer representing a TensorFlow sub-graph (specified by the input and output TensorFlow tensors).
It implements the standard BigDL layer API, and can be used with other Analytics-Zoo/BigDL layers
to construct more complex models for training or inference using the standard Analytics-Zoo/BigDL API. 

You can think of `TFDatasets`, `TFOptimizer`, `TFPredictor` as a set API for training/testing TensorFlow models
on Spark/BigDL, while `TFNet` as an Analytics-Zoo/BigDL layer initialized using TensorFlow graph.

For more information on TFNet, please refer to the [API Guide](../APIGuide/PipelineAPI/net.md##TFNet)

