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

## TFNet

TFNet is a analytics-zoo layer that wraps a tensorflow frozen graph and can easily run in parallel.

TFNet cannot be trained, so it can only be used for inference or as a feature extractor for fine tuning a model.
When used as feature extractor, there should not be any trainable layers before TFNet, as all the gradient
from TFNet is set to zero.

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
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

Please refer to [TFNet Object Detection Example (Scala)](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/tensorflow/tfnet)
or [TFNet Object Detection Example (Python)](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/tensorflow/tfnet) and
the [Image Classification Using TFNet Notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/tfnet) for more information.
