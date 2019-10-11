## TFDataset

TFDatset represents a distributed collection of elements to be feed into TensorFlow graph.
TFDatasets can be created using a RDD and each of its records is a list of numpy.ndarray representing
the tensors to be feed into TensorFlow graph on each iteration. TFDatasets must be used with the
TFOptimizer or TFPredictor.

__Note__: This feature currently requires __tensorflow 1.10__ and OS is one of the following 64-bit systems.
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.

To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

### Methods

#### from_rdd

Create a TFDataset from a rdd.

For training and evaluation, both `features` and `labels` arguments should be specified.
The element of the rdd should be a tuple of two, (features, labels), each has the
same structure of numpy.ndarrays of the argument `features`, `labels`.

E.g. if `features` is [(tf.float32, [10]), (tf.float32, [20])],
and `labels` is {"label1":(tf.float32, [10]), "label2": (tf.float32, [20])}
then a valid element of the rdd could be

        (
        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))],
         {"label1": np.zeros(dtype=float, shape=(10,)),
          "label2":np.zeros(dtype=float, shape=(10,))))}
        )

If `labels` is not specified,
then the above element should be changed to

        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))]

For inference, `labels` can be not specified.
The element of the rdd should be some ndarrays of the same structure of the `features`
argument.

A note on the legacy api: if you are using `names`, `shapes`, `types` arguments,
each element of the rdd should be a list of numpy.ndarray.

**Python**
```python
from_rdd(rdd, features, labels=None, batch_size=-1, batch_per_thread=-1, hard_code_batch_size=False, val_rdd=None)
```

**Arguments**

* **rdd**: a rdd containing the numpy.ndarrays to be used 
           for training/evaluation/inference
* **features**: the structure of input features, should one the following:

     - a tuple (dtype, shape), e.g. (tf.float32, [28, 28, 1]) 
     - a list of such tuple [(dtype1, shape1), (dtype2, shape2)],
                     e.g. [(tf.float32, [10]), (tf.float32, [20])],
     - a dict of such tuple, mapping string names to tuple {"name": (dtype, shape},
                     e.g. {"input1":(tf.float32, [10]), "input2": (tf.float32, [20])}
                    
* **labels**: the structure of input labels, format is the same as features
* **batch_size**: the batch size, used for training, should be a multiple of
        total core num
* **batch_per_thread**: the batch size for each thread, used for inference or evaluation
* **hard_code_batch_size**: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
* **val_rdd**: validation data with the same structure of rdd


#### from_ndarrays

Create a TFDataset from a nested structure of numpy ndarrays. Each element
in the resulting TFDataset has the same structure of the argument tensors and
is created by indexing on the first dimension of each ndarray in the tensors
argument.

This method is equivalent to sc.parallize the tensors and call TFDataset.from_rdd

**Python**
```python
from_ndarrays(tensors, batch_size=-1, batch_per_thread=-1, hard_code_batch_size=False, val_tensors=None)
```

**Arguments**

* **tensors**: the numpy ndarrays
* **batch_size**: the batch size, used for training, should be a multiple of
        total core num
* **batch_per_thread**: the batch size for each thread, used for inference or evaluation
* **hard_code_batch_size**: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
* **val_tensors**: the numpy ndarrays used for validation during training


#### from_image_set

Create a TFDataset from a ImagetSet. Each ImageFeature in the ImageSet should
already has the "sample" field, i.e. the result of ImageSetToSample transformer

**Python**
```python
from_image_set(image_set, image, label=None, batch_size=-1, batch_per_thread=-1, hard_code_batch_size=False, validation_image_set=None)
```

**Arguments**

* **image_set**: the ImageSet used to create this TFDataset
* **image**: a tuple of two, the first element is the type of image, the second element
        is the shape of this element, i.e. (tf.float32, [224, 224, 3]))
* **label**: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1]))
* **batch_size**: the batch size, used for training, should be a multiple of
        total core num
* **batch_per_thread**: the batch size for each thread, used for inference or evaluation
* **hard_code_batch_size**: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
* **validation_image_set**: the ImageSet used for validation during training


#### from_text_set

Create a TFDataset from a TextSet. The TextSet must be transformed to Sample, i.e.
the result of TextFeatureToSample transformer.

**Python**
```python
from_text_set(text_set, text, label=None, batch_size=-1, batch_per_thread=-1, hard_code_batch_size=False, validation_image_set=None)
```

**Arguments**

* **text_set**: the TextSet used to create this TFDataset
* **text**: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [10, 100, 4])).
        text can also be nested structure of this tuple of two.
* **label**: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
* **batch_size**: the batch size, used for training, should be a multiple of
        total core num
* **batch_per_thread**: the batch size for each thread, used for inference or evaluation
* **hard_code_batch_size**: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
* **validation_image_set**: The TextSet used for validation during training

#### from_feature_set

Create a TFDataset from a FeatureSet. Currently, the element in this Feature set must be a
ImageFeature that has a sample field, i.e. the result of ImageSetToSample transformer

**Python**
```python
from_feature_set(dataset, features, labels=None, batch_size=-1, batch_per_thread=-1, hard_code_batch_size=False, validation_dataset=None)
```

**Arguments**

* **dataset**: the feature set used to create this TFDataset
* **features**: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [224, 224, 3])).
        text can also be nested structure of this tuple of two.
* **labels**: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
* **batch_size**: the batch size, used for training, should be a multiple of
        total core num
* **batch_per_thread**: the batch size for each thread, used for inference or evaluation
* **hard_code_batch_size**: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
* **validation_dataset**: The FeatureSet used for validation during training