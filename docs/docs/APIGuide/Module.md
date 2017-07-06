---
## Model Save

BigDL supports saving models to local file system, HDFS and AWS S3. After a model is created, you can use `save` on created model to save it. Below example shows how to save a model. 

**Scala example**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

val model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train

model.save("/tmp/model.bigdl", true) //save to local fs
model.save("hdfs://...") //save to hdfs
model.save("s3://...") //save to s3

```
**Python example**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *

model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train
model.save("/tmp/model.bigdl", True) //save to local fs
model.save("hdfs://...") //save to hdfs
model.save("s3://...") //save to s3
```
In `model.save`, the first parameter is the path where we want to save our model, the second paramter is to specify if we need to overwrite the file if it already exists, it's set to false by default


## Model Load

Use `Module.load`(in Scala) or `Model.load` (in Python) to load an existing model.  `Module` (Scala) or `Model`(Python) is a utilily class provided in BigDL. We just need to specify the model path where we previously saved the model to load it to memory for resume training or prediction purpose.

**Scala example**
```scala
val model = Module.load("/tmp/model.bigdl") //load from local fs
val model = Module.load("hdfs://...") //load from hdfs
val model = Module.load("s3://...") //load from s3
```

**Python example**
```python
model = Model.load("/tmp/model.bigdl") //load from local fs
model = Model.load("hdfs://...") //load from hdfs
model = Model.load("s3://...") //load from s3
```

## Model Evaluation
**Scala**
```scala
model.evaluate(dataset,vMethods,batchSize = None)
```
**Python**
```python
model.test(val_rdd, batch_size, val_methods)
```

Use `evaluate` on the model for evaluation. The parameter `dataset` (Scala) or `val_rdd` (Python) in is the validation dataset, and `vMethods` (Scala) or `val_methods`(Python) is an array of ValidationMethods. Refer to [Metrics](Metrics.md) for the list of defined ValidationMethods. 

**Scala example**
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor

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

samples=[Sample.from_ndarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), np.array([2.0]))]
testSet = sc.parallelize(samples)

//train a model or load an existing model...
//model = ...
evaluateResult = model.test(testSet, 1, [Top1Accuracy])
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
Use `predict` or `predictClass` or `predict_class` on model for Prediction. `predict` returns return the probability distribution of each class, and `predictClass`/`predict_class` returns the predict label. They both accepts the test dataset as parameter. 

**Scala example**
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor

//create some dummy dataset for prediction as example
val feature = Tensor(10).rand()
val label = Tensor(1).randn()
val predictSample = Sample(feature, label)
val predictSet = sc.parallelize(Seq(predictSample))

//train a new model or load an existing model
//val model=... 
val preductResult = model.predict(predictSet)
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
 preductResult = model.predict(predictSet)
```

## Module Freeze
To "freeze" a module means to exclude some part of model from training.
Note that it only takes effect in BigDL graph model.

```scala
layer.setTrainable(false)
model.setFreeze(Array("layer1", "layer2"))
model.setTrainable(Array("layer3", "layer4"))
model.unFreeze()
```
* A single layer can be "freezed" by calling ```setTrainable```
* User can set freeze of list of layers in model by calling ```setFreeze```
* User can set a list of trainable layers and keep the remaining layers freezed by calling ```setTrainable```
* User can unfreeze all layers by calling ```unFreeze```
**Python**
```python
layer.set_trainable(false)
model.set_freeze(["layer1", "layer2"])
model.set_trainable(["layer1", "layer2"])
model.unfreeze()
```

**Scala**
```scala
val reshape = Reshape(Array(4)).inputs()
val fc1_1 = Linear(4, 2).setName("fc1_1").inputs()
val fc2_1 = Linear(4, 2).setName("fc2_1").setTrainable(false).inputs(reshape)
val cadd_1 = CAddTable().inputs(fc1_1, fc2_1)
val output1_1 = ReLU().inputs(cadd_1)
val output2_1 = Threshold(10.0).inputs(cadd_1)

val model = Graph(Array(reshape, fc1_1), Array(output1_1, output2_1))

fc1_1.element.getParameters()._1.apply1(_ => 1.0f)
fc2_1.element.getParameters()._1.apply1(_ => 2.0f)

val input = T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
  Tensor(T(0.5f, 0.4f, -0.2f, -0.1f)))
val gradOutput = T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f)))

println("output1: \n", model.forward(input))
println("gradInput1: \n", model.backward(input, gradOutput))

model.unFreeze()
println("output2: \n", model.forward(input))
println("gradInput2: \n", model.backward(input, gradOutput))

model.setFreeze(Array("fc2_1"))
println("output3: \n", model.forward(input))
println("gradInput3: \n", model.backward(input, gradOutput))

model.unFreeze()
model.setTrainable(Array("fc1_1"))
println("output4: \n", model.forward(input))
println("gradInput4: \n", model.backward(input, gradOutput))
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
(gradInput1: 
, {
	2: 3.0
	   3.0
	   3.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
	1: [com.intel.analytics.bigdl.tensor.DenseTensor with no dimension]
 })
(output2: 
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 3.6
	   3.6
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(gradInput2: 
, {
	2: 3.0
	   3.0
	   3.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
	1: 6.0
	   6.0
	   6.0
	   6.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
 })
(output3: 
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 3.6
	   3.6
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(gradInput3: 
, {
	2: 3.0
	   3.0
	   3.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
	1: [com.intel.analytics.bigdl.tensor.DenseTensor with no dimension]
 })
(output4: 
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 3.6
	   3.6
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(gradInput4: 
, {
	2: 3.0
	   3.0
	   3.0
	   3.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 4]
	1: [com.intel.analytics.bigdl.tensor.DenseTensor with no dimension]
 })
```



**Python**
```python
from bigdl.nn.layer import *
import numpy as np

reshape = Reshape([4])()
fc1_1 = Linear(4, 2).set_name("fc1_1")()
# set fc2_1 to non-trainable
fc2_1 = Linear(4, 2).set_name("fc2_1").set_trainable(False)(reshape)
cadd_1 = CAddTable()([fc1_1, fc2_1])
output1_1 = ReLU()(cadd_1)
output2_1 = Threshold(10.0)(cadd_1)
model = Model([reshape, fc1_1], [output1_1, output2_1])

input = [
    np.array([0.1, 0.2, -0.3, -0.4]),
    np.array([0.5, 0.4, -0.2, -0.1])]
gradOutput = [
    np.array([1.0, 2.0]), np.array([3.0, 4.0])]

output = model.forward(input)
print "output1: \n", output
gradInput = model.backward(input, gradOutput)
print "gradInput1 \n", gradInput

# unfreeze all layers in model
model.unfreeze()
output = model.forward(input)
print "output2 \n", output
gradInput = model.backward(input, gradOutput)
print "gradInput2 \n", gradInput

# set fc2_1 to non-trainable in model API
m3 = model.set_freeze(["fc2_1"])
output = model.forward(input)
print "output3 \n", output
gradInput = model.backward(input, gradOutput)
print "gradInput3 \n", gradInput

model.unfreeze()
# set only fc1_1 to trainable in model API
m3 = model.set_trainable(["fc1_1"])
output = model.forward(input)
print "output4 \n", output
gradInput = model.backward(input, gradOutput)
print "gradInput4 \n", gradInput
```

```
output1: 
[array([ 0.        ,  0.58186555], dtype=float32), array([ 0.,  0.], dtype=float32)]
gradInput1 
[array([], dtype=float32), array([ 0.510382  ,  0.85159016, -0.55450386, -0.54459006], dtype=float32)]
output2 
[array([ 0.        ,  0.58186555], dtype=float32), array([ 0.,  0.], dtype=float32)]
gradInput2 
[array([ 0.04848308, -0.04261303,  0.04275261, -0.94517899], dtype=float32), array([ 0.510382  ,  0.85159016, -0.55450386, -0.54459006], dtype=float32)]
output3 
[array([ 0.        ,  0.58186555], dtype=float32), array([ 0.,  0.], dtype=float32)]
gradInput3 
[array([], dtype=float32), array([ 0.510382  ,  0.85159016, -0.55450386, -0.54459006], dtype=float32)]
output4 
[array([ 0.        ,  0.58186555], dtype=float32), array([ 0.,  0.], dtype=float32)]
gradInput4 
[array([], dtype=float32), array([ 0.510382  ,  0.85159016, -0.55450386, -0.54459006], dtype=float32)]
```


