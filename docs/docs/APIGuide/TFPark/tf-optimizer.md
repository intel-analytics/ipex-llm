TFOptimizer is used for optimizing a TensorFlow model with respect to its training variables
on Spark/BigDL.

__Note__: This feature currently requires __tensorflow 1.10__ and OS is one of the following 64-bit systems.
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.

To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

**Create a TFOptimizer**:
```python
import tensorflow as tf
from zoo.tfpark import TFOptimizer
from bigdl.optim.optimizer import *
loss = ...
optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

For Keras model:

```python
from zoo.tfpark import TFOptimizer
from bigdl.optim.optimizer import *
from tensorflow.keras.models import Model

model = Model(inputs=..., outputs=...)

model.compile(optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

optimizer = TFOptimizer.from_keras(model, dataset)
optimizer.optimize(end_trigger=MaxEpoch(5))
```

## Methods

### from_loss (factory method)

Create a TFOptimizer from a TensorFlow loss tensor.
The loss tensor must come from a TensorFlow graph that only takes TFDataset.tensors and
the tensors in `tensor_with_value` as inputs.

```python
from_loss(loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0,
                  clip_norm=None, clip_value=None, metrics=None,
                  tensor_with_value=None, **kwargs)
```

#### Arguments


* **loss**: The loss tensor of the TensorFlow model, should be a scalar.
            The loss tensor must come from a TensorFlow graph that only takes TFDataset.tensors and
            the tensors in `tensor_with_value` as inputs.
* **optim_method**: the optimization method to be used, such as bigdl.optim.optimizer.Adam
* **session**: the current TensorFlow Session, if you want to used a pre-trained model,
             you should use the Session to load the pre-trained variables and pass it to TFOptimizer.
* **val_outputs**: the validation output TensorFlow tensor to be used by val_methods
* **val_labels**: the validation label TensorFlow tensor to be used by val_methods
* **val_method**: the BigDL val_method(s) to be used.
* **val_split**: Float between 0 and 1. Fraction of the training data to be used as
               validation data. 
* **clip_norm**: float >= 0. Gradients will be clipped when their L2 norm exceeds
               this value.
* **clip_value**: float >= 0. Gradients will be clipped when their absolute value
                exceeds this value.
* **metrics**: a dictionary. The key should be a string representing the metric's name
             and the value should be the corresponding TensorFlow tensor, which should be a scalar.
* **tensor_with_value**: a dictionary. The key is TensorFlow tensor, usually a
                      placeholder, the value of the dictionary is a tuple of two elements. The first one of
                      the tuple is the value to feed to the tensor in training phase and the second one
                      is the value to feed to the tensor in validation phase.


### from_keras (factory method)

Create a TFOptimizer from a tensorflow.keras model. The model must be compiled.

```python
from_keras(keras_model, dataset, optim_method=None, val_spilt=0.0, **kwargs)
```

#### Arguments

* **keras_model**: the tensorflow.keras model, which must be compiled.
* **dataset**: a [TFDataset](./tf-dataset.md)
* **optim_method**: the optimization method to be used, such as bigdl.optim.optimizer.Adam
* **val_spilt**: Float between 0 and 1. Fraction of the training data to be used as
      validation data.


### set_train_summary

```python
set_train_summary(summary)
```

#### Arguments

* **summary**: The train summary to be set. A TrainSummary object contains information
               necessary for the optimizer to know how often the logs are recorded,
               where to store the logs and how to retrieve them, etc. For details,
               refer to the docs of [TrainSummary](https://bigdl-project.github.io/0.9.0/#ProgrammingGuide/visualization/).

### set_val_summary

```python
set_val_summary(summary)
```

#### Arguments

* **summary**: The validation summary to be set. A ValidationSummary object contains information
               necessary for the optimizer to know how often the logs are recorded,
               where to store the logs and how to retrieve them, etc. For details,
               refer to the docs of [ValidationSummary](https://bigdl-project.github.io/0.9.0/#ProgrammingGuide/visualization/).
               

### set_constant_gradient_clipping

```python
set_constant_gradient_clipping(min_value, max_value)
```

#### Arguments

* **min_value**: the minimum value to clip by
* **max_value**: the maxmimum value to clip by


### set_gradient_clipping_by_l2_norm

```python
set_gradient_clipping_by_l2_norm(self, clip_norm)
```

#### Arguments

* **clip_norm**: gradient L2-Norm threshold


### optimize

```python
optimize(self, end_trigger=None)
```

#### Arguments

* **end_trigger**: BigDL's [Trigger](https://bigdl-project.github.io/0.9.0/#APIGuide/Triggers/) to indicate when to stop the training. If none, defaults to
                   train for one epoch.