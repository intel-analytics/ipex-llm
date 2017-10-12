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
model.evaluate(dataset, vMethods, batchSize = None)
```
**Python**
```python
model.evaluate(val_rdd, batch_size, val_methods)
```

Use `evaluate` on the model for evaluation. The parameter `dataset` (Scala) or `val_rdd` (Python) is the validation dataset, and `vMethods` (Scala) or `val_methods`(Python) is an array of ValidationMethods. Refer to [Metrics](Metrics.md) for the list of defined ValidationMethods.

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
To "freeze" a module means to exclude some layers of model from training.

```scala
layer.freeze()
layer.unFreeze()
model.freeze(Array("layer1", "layer2"))
model.unFreeze()
model.stopGradient(Array("layer1"))
```
* A single layer can be "freezed" by calling ```freeze()```. If a layer is freezed,
its parameters(weight/bias, if exists) are not changed in training process
* A single layer can be "unFreezed" by calling ```unFreeze()```.
* User can set freeze of list of layers in model by calling ```freeze```
* User can unfreeze all layers by calling ```unFreeze```
* stop the input gradient of layers that match the given names. Their input gradient are not computed.
And they will not contributed to the input gradient computation of layers that depend on them.

Note that stopGradient is only supported in Graph model.

**Python**
```python
layer.freeze()
layer.unfreeze()
model.freeze(["layer1", "layer2"])
model.unfreeze()
model.stop_gradient(["layer1"])
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
model.freeze(Array("fc2"))
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
