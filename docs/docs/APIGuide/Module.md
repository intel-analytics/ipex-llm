---
## Model Save

BigDL supports saving models to local file system, HDFS and AWS S3. After a model is created, you can use `saveModule` (Scala) or 'saveModel' (python) on created model to save it. Below example shows how to save a model.

**Scala example**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

val model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train

model.saveModule("/tmp/model.bigdl", "/tmp/model.bin", true) //save to local fs
model.saveModule("hdfs://...") //save to hdfs
model.saveModule("s3://...") //save to s3

```
**Python example**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *

model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train
model.saveModel("/tmp/model.bigdl", "/tmp/model.bin", True) //save to local fs
model.saveModel("hdfs://...") //save to hdfs
model.saveModel("s3://...") //save to s3
```
In `model.saveModel`, the first parameter is the path where we want to save our model network, the second parameter is the path where we want to save the model weights, the third parameter is to specify if we need to overwrite the file if it already exists, it's set to false by default
Please notice that if the second parameter is not specified, weights will be saved into the same file as model network. Save weights separately usually handles the situation that the model is big in size


## Model Load

### Load BigDL model

Use `Module.loadModule`(in Scala) or `Model.loadModel` (in Python) to load an existing model.  `Module` (Scala) or `Model`(Python) is a utility class provided in BigDL. We just need to specify the model path and optionally weight path if exists where we previously saved the model to load it to memory for resume training or prediction purpose.

**Scala example**
```scala
val model = Module.loadModule("/tmp/model.bigdl", "/tmp/model.bin") //load from local fs
val model = Module.loadModule("hdfs://...") //load from hdfs
val model = Module.loadModule("s3://...") //load from s3
```

**Python example**
```python
model = Model.loadModel("/tmp/model.bigdl", "/tmp/model.bin") //load from local fs
model = Model.loadModel("hdfs://...") //load from hdfs
model = Model.loadModel("s3://...") //load from s3
```

### Load Tensorflow model

BigDL also provides utilities to load tensorflow model. See [tensorflow support](https://bigdl-project.github.io/master/#ProgrammingGuide/tensorflow-support/)
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
val model = Module.loadTF(modelPath, inputs, outputs, ByteOrder.LITTLE_ENDIAN)
                            
// For tensorflow graph with Variables
val model = Module.loadTF(modelPath, inputs, outputs, ByteOrder.LITTLE_ENDIAN, Some(binPath))

// For tensorflow model inference only
val model = Module.loadTF(modelPath, inputs, outputs, ByteOrder.LITTLE_ENDIAN, generateBackward=false)
```

**Python example**
```python
model_def = "/tmp/model/model.pb"
model_variable = "/tmp/model/model.bin"
inputs = ["Placeholder"]
outputs = ["output"]
# For tensorflow freezed graph or graph without Variables
model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float")

# For tensorflow graph with Variables
model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)

# For tensorflow model inference only
model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", generated_backward=False, bigdl_type="float")
```

### Load Keras model

For __Python__ users, BigDL also supports loading pre-defined Keras models. See [keras support](../ProgrammingGuide/keras-support.md) for more details.

Note that the Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

A Keras model definition in __JSON__ file can be loaded as a BigDL model.
Saved weights in __HDF5__ file can also be loaded together with the architecture of a Keras model.

You can directly call the API `Model.load_keras` to load a Keras model into BigDL.

__Remark__: `keras==1.2.2` is required. If you need to load a HDF5 file, you also need to install `h5py`. These packages can be installed via `pip` easily.

```python
from bigdl.nn.layer import *

bigdl_model = Model.load_keras(json_path=None, hdf5_path=None, by_name=False)
```

## Model Evaluation
**Scala**
```scala
model.evaluate(dataset, vMethods, batchSize = None)
```
**Python**
```python
model.evaluate(val_rdd, batch_size, val_methods)
```

Use `evaluate` on the model for evaluation. The parameter `dataset` (Scala) or `val_rdd` (Python) is the validation dataset, and `vMethods` (Scala) or `val_methods`(Python) is an array of ValidationMethods. Refer to [Metrics](Metrics.md) for the list of defined ValidationMethods.

For most CNN models, it's recommended to enable MKL-DNN acceleration by specifying `bigdl.engineType` as `mkldnn` for evaluation.

**Scala example**
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.utils.engine
import org.apache.spark.SparkContext

val conf = Engine.createSparkConf()
    .setAppName("Model")
    .setMaster("local[*]")
val sc = new SparkContext(conf)
    Engine.init

//create some dummy dataset for evaluation
val feature = Tensor(10).rand()
val label = Tensor(1).randn()

val testSample = Sample(feature, label)
//sc is is the SparkContxt instance
val testSet = sc.parallelize(Seq(testSample))

//train a new model or load an existing model
//val model=...
val evaluateResult = model.evaluate(testSet, Array(new Top1Accuracy))
```

**Python example**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *
import numpy as np

sc = SparkContext.getOrCreate(conf=create_spark_conf())
init_engine()

samples=[Sample.from_ndarray(np.array([1.0, 2.0]), np.array([2.0]))]
testSet = sc.parallelize(samples,1)

//You can train a model or load an existing model before evaluation.
model = Linear(2, 1)

evaluateResult = model.evaluate(testSet, 1, [Top1Accuracy()])
print(evaluateResult[0])
```


## Model Prediction
**Scala**
```scala
model.predict(dataset)
model.predictClass(dataset)
```

**Python**
```python
model.predict(data_rdd)
model.predict_class(data_rdd)
```
Use `predict` or `predictClass` or `predict_class` on model for Prediction. `predict` returns the probability distribution of each class, and `predictClass`/`predict_class` returns the predict label. They both accepts the test dataset as parameter.

Please note that the sequence and the partitions of the output rdd will keep the same with input. So you can zip the output rdd with input rdd to get a (data, result) pair rdd.

For most CNN models, it's recommended to enable MKL-DNN acceleration by specifying `bigdl.engineType` as `mkldnn` for prediction.

**Scala example**
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor

//create some dummy dataset for prediction as example
val feature = Tensor(10).rand()
val predictSample = Sample(feature)
val predictSet = sc.parallelize(Seq(predictSample))

//train a new model or load an existing model
//val model=...
val predictResult = model.predict(predictSet)
```

**Python example**
```python
 from bigdl.nn.layer import *
 from bigdl.util.common import *
 from bigdl.optim.optimizer import *
 import numpy as np

 samples=[Sample.from_ndarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), np.  array([2.0]))]

 predictSet = sc.parallelize(samples)

 //train a model or load an existing model...
 //model = ...
 predictResult = model.predict(predictSet)
```

## Module Freeze
To "freeze" a module means to exclude some layers of model from training.

```scala
module.freeze("layer1", "layer2")
module.unFreeze("layer1", "layer2")
module.stopGradient(Array("layer1"))
```
* The whole module can be "freezed" by calling ```freeze()```. If a module is freezed,
its parameters(weight/bias, if exists) are not changed in training process.
If module names are passed, then layers that match the given names will be freezed.
* The whole module can be "unFreezed" by calling ```unFreeze()```.
If module names are provided, then layers that match the given names will be unFreezed.
* stop the input gradient of layers that match the given names. Their input gradient are not computed.
And they will not contributed to the input gradient computation of layers that depend on them.

Note that stopGradient is only supported in Graph model.

**Python**
```python
module.freeze(["layer1", "layer2"])
module.unfreeze(["layer1", "layer2"])
module.stop_gradient(["layer1"])
```

**Scala**
Original model without "freeze" or "stop gradient"
```scala
val reshape = Reshape(Array(4)).inputs()
val fc1 = Linear(4, 2).setName("fc1").inputs()
val fc2 = Linear(4, 2).setName("fc2").inputs(reshape)
val cadd_1 = CAddTable().setName("cadd").inputs(fc1, fc2)
val output1_1 = ReLU().inputs(cadd_1)
val output2_1 = Threshold(10.0).inputs(cadd_1)

val model = Graph(Array(reshape, fc1), Array(output1_1, output2_1))

val input = T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
  Tensor(T(0.5f, 0.4f, -0.2f, -0.1f)))
val gradOutput = T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f)))

fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
println("output1: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```
```
(output1:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,1.9	1.8	2.3	2.4
1.8	1.6	2.6	2.8
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```

"Freeze" ```fc2```, the parameters of ```fc2``` is not changed.
```scala
fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
model.freeze("fc2")
println("output2: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```

```
(output2:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,2.0	2.0	2.0	2.0
2.0	2.0	2.0	2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```
"unFreeze" ```fc2```, the parameters of ```fc2``` will be updated.
```scala
fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
model.unFreeze()
println("output3: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```
```
(output3:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,1.9	1.8	2.3	2.4
1.8	1.6	2.6	2.8
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```

"stop gradient" at ```cadd```, the parameters of ```fc1``` and ```fc2``` are not changed.
```scala
fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.stopGradient(Array("cadd"))
model.zeroGradParameters()
println("output4: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc1 weight \n", fc1.element.parameters()._1(0))
println("fc2 weight \n", fc2.element.parameters()._1(0))
```

```
(output4:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc1 weight
,1.0	1.0	1.0	1.0
1.0	1.0	1.0	1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
(fc2 weight
,2.0	2.0	2.0	2.0
2.0	2.0	2.0	2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```



**Python**
```python
from bigdl.nn.layer import *
import numpy as np

reshape = Reshape([4])()
fc1 = Linear(4, 2).set_name("fc1")()
fc2 = Linear(4, 2).set_name("fc2")(reshape)
cadd = CAddTable().set_name("cadd")([fc1, fc2])
output1 = ReLU()(cadd)
output2 = Threshold(10.0)(cadd)
model = Model([reshape, fc1], [output1, output2])

input = [
    np.array([0.1, 0.2, -0.3, -0.4]),
    np.array([0.5, 0.4, -0.2, -0.1])]
gradOutput = [
    np.array([1.0, 2.0]), np.array([3.0, 4.0])]

fc1.element().set_weights([np.array([[1,1,1,1],[1,1,1,1]]), np.array([1,1])])
fc2.element().set_weights([np.array([[2,2,2,2],[2,2,2,2]]), np.array([2,2])])
model.zero_grad_parameters()
output = model.forward(input)
print "output1: ", output
gradInput = model.backward(input, gradOutput)
model.update_parameters(1.0)
print "fc2 weight \n", fc2.element().parameters()['fc2']['weight']
```

```
> output1
[array([ 2.79999995,  2.79999995], dtype=float32), array([ 0.,  0.], dtype=float32)]

> fc2 weight
[[ 1.89999998  1.79999995  2.29999995  2.4000001 ]
 [ 1.79999995  1.60000002  2.5999999   2.79999995]]
```

```
fc1.element().set_weights([np.array([[1,1,1,1],[1,1,1,1]]), np.array([1,1])])
fc2.element().set_weights([np.array([[2,2,2,2],[2,2,2,2]]), np.array([2,2])])
m3 = model.freeze(["fc2"])
model.zero_grad_parameters()
output = model.forward(input)
print "output2 ", output
gradInput = model.backward(input, gradOutput)
model.update_parameters(1.0)
print "fc2 weight \n", fc2.element().parameters()['fc2']['weight']
```

```
> output2
[array([ 2.79999995,  2.79999995], dtype=float32), array([ 0.,  0.], dtype=float32)]

> fc2 weight
[[ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]]
```

```
fc1.element().set_weights([np.array([[1,1,1,1],[1,1,1,1]]), np.array([1,1])])
fc2.element().set_weights([np.array([[2,2,2,2],[2,2,2,2]]), np.array([2,2])])
m3 = model.unfreeze()
model.zero_grad_parameters()
output = model.forward(input)
print "output3 ", output
gradInput = model.backward(input, gradOutput)
model.update_parameters(1.0)
print "fc2 weight \n", fc2.element().parameters()['fc2']['weight']
```

```
> output3
[array([ 2.79999995,  2.79999995], dtype=float32), array([ 0.,  0.], dtype=float32)]

> fc2 weight
[[ 1.89999998  1.79999995  2.29999995  2.4000001 ]
 [ 1.79999995  1.60000002  2.5999999   2.79999995]]
```

```
m3 = model.stop_gradient(["cadd"])
model.zero_grad_parameters()
output = model.forward(input)
print "output4 ", output
gradInput = model.backward(input, gradOutput)
model.update_parameters(1.0)
print "fc1 weight \n", fc1.element().parameters()['fc1']['weight']
print "fc2 weight \n", fc2.element().parameters()['fc2']['weight']
```

```
> output4
[array([ 2.79999995,  2.79999995], dtype=float32), array([ 0.,  0.], dtype=float32)]

> fc1 weight
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]

> fc2 weight
[[ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]]
```
## Caffe Model Support
### Load Caffe model

**Scala:**

```scala
Module.loadCaffeModel(defPath, modelPath)
```
**Python:**
```python
Model.load_caffe_model(defPath, modelPath)
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Module.loadCaffeModel("/tmp/deploy.prototxt", "/tmp/caffe.caffemodel")
```

In above `defPath` specifies the path for the network deploy file while `modelPath` specifies the path for the weight file 

**Python example:**

``` python
from bigdl.nn.layer import *
model = Model.load_caffe_model("/tmp/deploy.prototxt", "/tmp/caffe.caffemodel")
```

### Load weight from Caffe into pre-defined model

**Scala:**

```scala
Module.loadCaffe(model, defPath, modelPath, match_all = true)
```
**Python:**
```python
Model.load_caffe(model, defPath, modelPath, match_all = True)
```

`model` is pre-defined BigDL model. Similar to `loadCaffeModel`, `defPath` and `modelPath` specify network deploy file and weight file,
the 4th parameter `match_all` specifies if layer definition should be exactly matched between pre-defined `model` and the one from `defPath`

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Sequential().add(Linear(3, 4))
val loadedModel = Module.loadCaffe(model, "/tmp/deploy.prototxt", "/tmp/caffe.caffemodel", true)
```


**Python example:**

``` python
from bigdl.nn.layer import *
model = Sequential().add(Linear(3, 4))
loadedModel = Model.load_caffe(model, "/tmp/deploy.prototxt", "/tmp/caffe.caffemodel", True)
```

### Save BigDL model as Caffe model

**Scala:**

```scala
bigdlModel.saveCaffe(prototxtPath, modelPath, useV2 = true, overwrite = false)
```
**Python:**
```python
bigdl_model.save_caffe(prototxt_path, model_path, use_v2 = True, overwrite = False)
```

`prototxtPath` defines where to store the network, `modelPath` defines where to store the weight, `useV2` 
defines whether to store as V2Layer format, and `overwrite` defines whether to overwrite if the files already exist.

Only Graph model is supported for now.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
val linear = Linear(3, 4)
val model = Graph(linear.inputs(), linear.inputs())
model.saveCaffe("/tmp/linear.prototxt", "/tmp/linear.caffemodel", true, true)
```

**Python example:**

``` python
from bigdl.nn.layer import *
linear = Linear(3, 4)
model = Graph(linear.inputs(), linear.inputs())
model.save_caffe(model, "/tmp/linear.prototxt", "/tmp/linear.caffemodel", True, True)
```
