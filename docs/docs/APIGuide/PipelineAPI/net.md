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

### Load Tensorflow model

We also provides utilities to load tensorflow model.
for more information.

If we already have a freezed graph protobuf file, we can use the `loadTF` api directly to
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

// For tensorflow freezed graph or graph without Variables
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
# For tensorflow freezed graph or graph without Variables
model = Net.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float")

# For tensorflow graph with Variables
model = Net.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)
```

## TFNet

**Scala:**
```scala
val m = TFNet(freezedModelPath, inputs, outputs)
```
**Python:**
```python
m = TFNet(freezed_model_path, inputs, outputs)
```

This is a layer that wraps a tensorflow freezed sub graph as a layer and run tensorflow in parallel.



