# KerasModel

KerasModel enables user to use `tf.keras` API to define TensorFlow models and perform training or evaluation on top
of Spark and BigDL in a distributed fashion.

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


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

## Methods

### \_\_init\_\_

```python
KerasModel(model)
```

#### Arguments

* **model**: a compiled keras model defined using `tf.keras`


### fit

```python
fit(x=None, y = None, batch_size=None, epochs=1, validation_data=None, distributed=False)
```

#### Arguments

* **x**: Input data. It could be:

         - a TFDataset object
         - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
         - A dict mapping input names to the corresponding array/tensors, if the model has named inputs.

* **y**: Target data. Like the input data `x`,
         It should be consistent with `x` (you cannot have Numpy inputs and
         tensor targets, or inversely). If `x` is a TFDataset, `y` should
         not be specified (since targets will be obtained from `x`).
         
* **batch_size**: Integer or `None`.
                  Number of samples per gradient update.
                  If `x` is a TFDataset, you do not need to specify batch_size.

* **epochs**: Integer. Number of epochs to train the model.
              An epoch is an iteration over the entire `x` and `y`
              data provided.

* **validation_data**: validation_data: Data on which to evaluate
                       the loss and any model metrics at the end of each epoch.
                       The model will not be trained on this data.
                       `validation_data` could be:
                          - tuple `(x_val, y_val)` of Numpy arrays or tensors

* **distributed**: Boolean. Whether to do prediction in distributed mode or local mode.
                   Default is True. In local mode, x must be a Numpy array.
                   
                   
### evaluate

```python
evaluate(x=None, y=None, bath_per_thread=None, distributed=False)
```

#### Arguments

* **x**: Input data. It could be:

            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
* **y**: Target data. Like the input data `x`,
         It should be consistent with `x` (you cannot have Numpy inputs and
         tensor targets, or inversely). If `x` is a TFDataset, `y` should
         not be specified (since targets will be obtained from `x`).
* **batch_per_thread**:
          The default value is 1.
          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
          When distributed is False the total batch size is batch_per_thread * numOfCores.
* **distributed**: Boolean. Whether to do prediction in distributed mode or local mode.
                   Default is True. In local mode, x must be a Numpy array.


### predict

```python
predict(x, batch_per_thread=None, distributed=False):
```

#### Arguments
* **x**: Input data. It could be:

            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,
* **batch_per_thread**:
          The default value is 1.
          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
          When distributed is False the total batch size is batch_per_thread * numOfCores.
* **distributed**: Boolean. Whether to do prediction in distributed mode or local mode.
                    Default is True. In local mode, x must be a Numpy array.




