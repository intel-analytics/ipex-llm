## Net

### Load Analytics Zoo Model

Use `Net.load`(in Scala) or `Net.load` (in Python) to load an existing model defined using the Analytics Zoo Keras-style API.  `Net` (Scala) or `Net`(Python) is a utility class provided in Analytics Zoo. We just need to specify the model path and optionally weight path if exists where we previously saved the model to load it to memory for resume training or prediction purpose.

**Scala example**
```scala
val model = Net.load("/tmp/model.def", "/tmp/model.weights") //load from local fs
val model = Net.load("hdfs://...") //load from hdfs
val model = Net.load("s3://...") //load from s3
```

**Python example**
```python
model = Net.load("/tmp/model.def", "/tmp/model.weights") //load from local fs
model = Net.load("hdfs://...") //load from hdfs
model = Net.load("s3://...") //load from s3
```

### Load BigDL Model

**Scala example**
```scala
val model = Net.loadBigDL("/tmp/model.def", "/tmp/model.weights") //load from local fs
val model = Net.loadBigDL("hdfs://...") //load from hdfs
val model = Net.loadBigDL("s3://...") //load from s3
```

**Python example**
```python
model = Net.loadBigDL("/tmp/model.def", "/tmp/model.weights") //load from local fs
model = Net.loadBigDL("hdfs://...") //load from hdfs
model = Net.loadBigDL("s3://...") //load from s3
```

### Load Torch Model

**Scala example**
```scala
val model = Net.loadTorch("/tmp/torch_model") //load from local fs
val model = Net.loadTorch("hdfs://...") //load from hdfs
val model = Net.loadTorch("s3://...") //load from s3
```

**Python example**
```python
model = Net.loadTorch("/tmp/torch_model") //load from local fs
model = Net.loadTorch("hdfs://...") //load from hdfs
model = Net.loadTorch("s3://...") //load from s3
```

### Load Caffe Model

**Scala example**
```scala
val model = Net.loadCaffe("/tmp/def/path", "/tmp/model/path") //load from local fs
val model = Net.loadCaffe("hdfs://def/path", "hdfs://model/path") //load from hdfs
val model = Net.loadCaffe("s3://def/path", "s3://model/path") //load from s3
```

**Python example**
```python
model = Net.loadCaffe("/tmp/def/path", "/tmp/model/path") //load from local fs
model = Net.loadCaffe("hdfs://def/path", "hdfs://model/path") //load from hdfs
model = Net.loadCaffe("s3://def/path", "s3://model/path") //load from s3
```

### Load TensorFlow model

We also provides utilities to load tensorflow model.

If we already have a frozen graph protobuf file, we can use the `loadTF` api directly to
load the tensorflow model. 

Otherwise, we should first use the `export_tf_checkpoint.py` script provided by BigDL's distribution
package, or the `dump_model` function defined in [here](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/util/tf_utils.py) to
generate the model definition file (`model.pb`) and variable binary file (`model.bin`). 

**Use Script**
```shell
GRAPH_META_FILE=/tmp/tensorflow/model.ckpt.meta
CKPT_FILE_PREFIX=/tmp/tensorflow/model.ckpt
SAVE_PATH=/tmp/model/
python export_tf_checkpoint.py $GRAPH_META_FILE $CKPT_FILE_PREFIX $SAVE_PATH
```

**Use python function**
```python
import tensorflow as tf

# This is your model definition.
xs = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.zeros([1,10])+0.2)
b1 = tf.Variable(tf.zeros([10])+0.1)
Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs,W1), b1)
output = tf.nn.tanh(Wx_plus_b1, name="output")

# Adding the following lines right after your model definition 
from bigdl.util.tf_utils import dump_model
dump_model_path = "/tmp/model"
# This line of code will create a Session and initialized all the Variable and
# save the model definition and variable to dump_model_path as BigDL readable format.
dump_model(path=dump_model_path)
```

Then we can use the `loadTF` api to load the tensorflow model into BigDL.

**Scala example**
```scala
val modelPath = "/tmp/model/model.pb"
val binPath = "/tmp/model/model.bin"
val inputs = Seq("Placeholder")
val outputs = Seq("output")

// For tensorflow frozen graph or graph without Variables
val model = Net.loadTF(modelPath, inputs, outputs, ByteOrder.LITTLE_ENDIAN)
                            
// For tensorflow graph with Variables
val model = Net.loadTF(modelPath, inputs, outputs, ByteOrder.LITTLE_ENDIAN, Some(binPath))
```

**Python example**
```python
model_def = "/tmp/model/model.pb"
model_variable = "/tmp/model/model.bin"
inputs = ["Placeholder"]
outputs = ["output"]
# For tensorflow frozen graph or graph without Variables
model = Net.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float")

# For tensorflow graph with Variables
model = Net.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)
```

## TFNet

TFNet is a analytics-zoo layer that wraps a tensorflow frozen graph and can easily run in parallel.

The difference between Net.loadTF() is that TFNet will call tensorflow's java api to do the computation.

TFNet cannot be trained, so it can only be used for inference or as a feature extractor for fine tuning a model.
When used as feature extractor, there should not be any trainable layers before TFNet, as all the gradient
from TFNet is set to zero.

__Note__: This feature currently supports __tensorflow 1.10__ and requires the OS to be one of the following 64-bit systems.
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.

To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


### Export TensorFlow model to frozen inference graph

Analytics-zoo provides a useful utility function, `export_tf`, to export a TensorFlow model
to frozen inference graph.

For example:

**Python:**

```python
import tensorflow as tf
from nets import inception
slim = tf.contrib.slim

images = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))

with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits, end_points = inception.inception_v1(images, num_classes=1001, is_training=False)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "/tmp/models/inception_v1.ckpt")

from zoo.util.tf import export_tf
export_tf(sess, "/tmp/models/tfnet", inputs=[images], outputs=[logits])
```

In the above code, the `export_tf` utility function will frozen the TensorFlow graph, strip unused operation according to the inputs and outputs and save it to the specified directory along with the input/output tensor names. 


### Creating a TFNet

After we have export the TensorFlow model, we can easily create a TFNet.

**Scala:**
```scala
val m = TFNet("/tmp/models/tfnet")
```
**Python:**
```python
m = TFNet.from_export_folder("/tmp/models/tfnet")
```

Please refer to [TFNet Object Detection Example (Scala)](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/tfnet)
or [TFNet Object Detection Example (Python)](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/tensorflow/tfnet) and
the [Image Classification Using TFNet Notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/tfnet) for more information.


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

## TFOptimizer
TFOptimizer is the class that does all the hard work in distributed training, such as model
distribution and parameter synchronization. There are two ways to create a TFOptimizer.

The `from_loss` API takes the **loss** (a scalar tensor) as input and runs
stochastic gradient descent using the given **optimMethod** on all the **Variables** that contributing
to this loss.

The `from_keras` API takes a compiled **Keras Model** and a **TFDataset** and runs stochastic gradient
descent using the loss function, optimizer and metrics specified by the Keras model.

__Note__: This feature currently requires __tensorflow 1.10__ and OS is one of the following 64-bit systems.
__Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.

To run on other system may require you to manually compile the TensorFlow source code. Instructions can
be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).

**Python**
```python
loss = ...
optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
optimizer.optimize(end_trigger=MaxEpoch(5))
```

For Keras model:

```python


model = Model(inputs=..., outputs=...)

model.compile(optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

optimizer = TFOptimizer.from_keras(model, dataset)
optimizer.optimize(end_trigger=MaxEpoch(5))
```

## TFPredictor

TFPredictor takes a list of TensorFlow tensors as the model outputs and feed all the elements in
 TFDatasets to produce those outputs and returns a Spark RDD with each of its elements representing the
 model prediction for the corresponding input elements.
 
 __Note__: This feature currently requires __tensorflow 1.10__ and OS is one of the following 64-bit systems.
 __Ubuntu 16.04 or later__, __macOS 10.12.6 or later__ and __Windows 7 or later__.
 
 To run on other system may require you to manually compile the TensorFlow source code. Instructions can
 be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).
 
**Python**
```python
logist = ...
predictor = TFPredictor.from_outputs(sess, [logits])
predictions_rdd = predictor.predict()
```

For Keras model:
```python
model = Model(inputs=..., outputs=...)
model.load_weights("/tmp/mnist_keras.h5")
predictor = TFPredictor.from_keras(model, dataset)
predictions_rdd = predictor.predict()
```
