BigDL provide model API to faciliate user's requirement to save module to specific path, load module from given path, evaluate module and predict with module ,etc.

Here we introduce how to leverage this APIs


Suppose we have a sequential model defined as below : 

```
Linear -> Sigmoid -> Softmax
```

**Scala:**

Import necessary dependencies

```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor
```

###### Define the model

```scala
val model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
```

###### Save model to specific path

BigDL supports different file systems like Linux file system, HDFS and AWS S3, Assume we want to save to local file system
```scala
model.save("/tmp/model.bigdl", true)
```

In the above code , the first parameter is the path where we want to save our model, the second paramter is to specify if we need to overwrite the file if it already exists, it's set to false by default

###### Load model from given path

We just need to specify the model path where we previously saved the model to load it to memory for resume training or prediction purpose

```scala
val loadedModel = Module.load("/tmp/model.bigdl")
```
`Module` above is a utilily  to manipulate module APIs

###### Evaluate model with test set

Evaluation needs real test data set, here we just simulate some dummy data for function introduction purpose

Create dummy test data set

```scala
val feature = Tensor(10).apply1(e => Random.nextFloat())

val label = Tensor(1).apply1(e => Random.nextFloat())

val testSample = Sample(feature, label)
    
val testSet = sc.parallelize(Seq(testSample))
```
In above code, `sc` is the SparkContxt instance

Use the test dataset to evaluate the model

```scala
val evaluateResult = model.evaluate(testSet, Array(new Top1Accuracy))
```

###### Predict with model
Similar to above, we create input data first
```scala
val feature = Tensor(10).apply1(e => Random.nextFloat())

val label = Tensor(1).apply1(e => Random.nextFloat())

val predictSample = Sample(feature, label)
    
val predictSet = sc.parallelize(Seq(predictSample))
```

Predict with the data set
```scala
val preductResult = model.predict(predictSet)
```

**Python:**

```python
from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *
import numpy as np
```
###### Define the model
```python
model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())
```
###### Save model to specific path

```python
model.save("/tmp/model.bigdl", True)
```
###### Load model from given path

```python
model = Model.load("/tmp/model.bigdl")
```

`Model` is a utility for python mirroring `Module` in scala

###### Evaluate model with test set

Create dummy test data set 

```python
samples=[Sample.from_ndarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), np.array([2.0]))]

testSet = sc.parallelize(samples)

```
`np` above is numpy package

Use the test data to evaluate the model

```python
evaluateResult = model.evaluate(testSet, 1, [Top1Accuracy])
```

###### Predict with model

Create predict dummy data set
```python
samples=[Sample.from_ndarray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), np.array([2.0]))]

predictSet = sc.parallelize(samples)
```
Predict with the data set

```python
preductResult = model.predict(predictSet)
```
