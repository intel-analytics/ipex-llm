## **Loading a Tensorflow model into BigDL**

If you have a pre-trained Tensorflow model saved in a ".pb" file, you can load it
into BigDL.

For more information on how to generate
the ".pb" file, you can refer to [A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/extend/tool_developers/).
Specifically, you should generate a model definition file and a set of checkpoints, then use the [freeze_graph](https://github.com/tensorflow/tensorflow/blob/v1.0.0/tensorflow/python/tools/freeze_graph.py)
script to freeze the graph definition and weights in checkpoints into a single file.

### Generate model definition file and checkpoints in Tensorflow

**Python**
```python
import tensorflow as tf
xs = tf.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.zeros([1,10])+0.2)
b1 = tf.Variable(tf.zeros([10])+0.1)
Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs,W1), b1)
output = tf.nn.tanh(Wx_plus_b1, name="output")

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    checkpointpath = saver.save(sess, '/tmp/model/test.chkp')
    tf.train.write_graph(sess.graph, '/tmp/model', 'test.pbtxt')
```

### Freeze graph definition and checkpoints into a single ".pb" file

**Shell**
```shell
wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
python freeze_graph.py --input_graph /tmp/model/test.pbtxt --input_checkpoint /tmp/model/test.chkp --output_node_names=output --output_graph "/tmp/model/test.pb"
```

### Load Tensorflow model in BigDL

**Scala**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.Module
import java.nio.ByteOrder

val path = "/tmp/model/test.pb"
val inputs = Seq("Placeholder")
val outputs = Seq("output")
val model = Module.loadTF(path, Seq("Placeholder"), Seq("output"), ByteOrder.LITTLE_ENDIAN)
```

**Python**
```python
from bigdl.nn.layer import *
path = "/tmp/model/test.pb"
inputs = ["Placeholder"]
outputs = ["output"]
model = Model.load_tensorflow(path, inputs, outputs, byte_order = "little_endian", bigdl_type="float")
```
---

## **Saving a BigDL functional model to Tensorflow model file**

You can also save a [functional model](./Model/Functional.md) to protobuf files so that it can be used in Tensorflow inference.

When saving the model, placeholders will be added to the tf model as input nodes. So
you need to pass in the names and shapes of the placeholders. BigDL model does not have
such information. The order of the placeholder information should be same as the inputs
of the graph model.

**Scala**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.tf.TensorflowSaver
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
// create a graph model
val linear = Linear(10, 2).inputs()
val sigmoid = Sigmoid().inputs(linear)
val softmax = SoftMax().inputs(sigmoid)
val model = Graph(Array(linear), Array(softmax))

// save it to Tensorflow model file
model.saveTF(Seq(("input", Seq(4, 10))), "/tmp/model.pb")
```

**Python**
```python
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

# create a graph model
linear = Linear(10, 2)()
sigmoid = Sigmoid()(linear)
softmax = SoftMax()(sigmoid)
model = Model([linear], [softmax])

# save it to Tensorflow model file
model.save_tensorflow([("input", [4, 10])], "/tmp/model.pb")
```

---
## **Build Tensorflow model and run on BigDL**

You can construct your BigDL model directly from the input and output nodes of
Tensorflow model. That is to say, you can use Tensorflow to define
a model and use BigDL to run it.

**Python:**
```python
import tensorflow as tf
import numpy as np
from bigdl.nn.layer import *

tf.set_random_seed(1234)
input = tf.placeholder(tf.float32, [None, 5])
weight = tf.Variable(tf.random_uniform([5, 10]))
bias = tf.Variable(tf.random_uniform([10]))
middle = tf.nn.bias_add(tf.matmul(input, weight), bias)
output = tf.nn.tanh(middle)

# construct BigDL model and get the result form 
bigdl_model = Model(input, output, model_type="tensorflow")
```
