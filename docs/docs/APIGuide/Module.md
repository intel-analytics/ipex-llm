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

