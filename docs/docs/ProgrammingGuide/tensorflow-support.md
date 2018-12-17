BigDL supports loading and saving tensorflow models.
This page will give you a basic introduction of this feature. For more
interesting and sophisticated examples, please checkout [here](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/tensorflow).

## **Loading a Tensorflow model into BigDL**

BigDL supports loading tensorflow model with only a few lines of code.

If we already have a freezed graph protobuf file, we can use the `loadTF` api directly to
load the tensorflow model. 

Otherwise, we should first use the `export_tf_checkpoint.py` script provided by BigDL's distribution
package, or the `dump_model` function defined in [here](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/util/tf_utils.py) to
generate the model definition file (`model.pb`) and variable binary file (`model.bin`). 


### Generate model definition file and variable binary file


**Use Script**
```shell
CKPT_FILE_PREFIX=/tmp/tensorflow/model.ckpt
SAVE_PATH=/tmp/model/
python export_tf_checkpoint.py $CKPT_FILE_PREFIX $SAVE_PATH
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

Optionally, you can also pass in a initialized (either from scratch or from a checkpoint) Session object containing
all the variables of your model or pass in a pre-trained checkpoint path directly. See the `dump_model` doc in this
[file](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/util/tf_utils.py).

### Load Tensorflow model in BigDL

**Scala**
```scala
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.Module
import java.nio.ByteOrder

val modelPath = "/tmp/model/model.pb"
val binPath = "/tmp/model/model.bin"
val inputs = Seq("Placeholder")
val outputs = Seq("output")
val model = Module.loadTF(modelPath, Seq("Placeholder"),
    Seq("output"), ByteOrder.LITTLE_ENDIAN, Some(binPath))
```

**Python**
```python
from bigdl.nn.layer import *
model_def = "/tmp/model/model.pb"
model_variable = "/tmp/model/model.bin"
inputs = ["Placeholder"]
outputs = ["output"]
model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)
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

## **Supported Operations**
Please check this [page](../APIGuide/tensorflow_ops_list.md)
