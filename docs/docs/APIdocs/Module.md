## Module Class Overview
Module (Scala) or Model (Python) provides API to faciliate user's requirement to save model to specific path, load model from given path, evaluate model and predict with model, etc.


## Model Save

BigDL supports different file systems like Linux file system, HDFS and AWS S3. Use `model.save` to save models. Below is an example of how to save a model to local file system. 

**Scala example**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

val model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train
model.save("/tmp/model.bigdl", true)
```
**Python example**
```python
from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *

model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
//...train
model.save("/tmp/model.bigdl", True)
```
In `model.save`, the first parameter is the path where we want to save our model, the second paramter is to specify if we need to overwrite the file if it already exists, it's set to false by default


## Model Load

Use `Module.load`(in Scala) or `Model.load` (in Python) to load an existing model. We just need to specify the model path where we previously saved the model to load it to memory for resume training or prediction purpose

**Scala example**
```scala
val loadedModel = Module.load("/tmp/model.bigdl")
```
`Module` above is a utilily  to manipulate module APIs

**Python example**
```python
model = Model.load("/tmp/model.bigdl")
```
`Model` is a utility for python mirroring `Module` in scala

## Model Evaluation

Use `model.evaluate` to evaluate the model with validation data.

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
evaluateResult = model.evaluate(testSet, 1, [Top1Accuracy])
```


## Model Prediction

Use `model.predict` for Prediction.

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

