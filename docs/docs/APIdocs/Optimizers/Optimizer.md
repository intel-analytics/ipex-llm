## Optimizer ##

You can use Optimizer to distributed train your model with
a spark cluster.

### How to use Optimizer 
You need at least provide model, data, loss function and batch size.

* **model**

A neural network model. May be a layer, a sequence of layers or a
graph of layers.

* **data**

Your training data. As we train models on Spark, one of
the most common distributed data structures is RDD. Of course
you can use DataFrame. Please check the BigDL pipeline example.

```scala
def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      featurePaddingParam: PaddingParam[T],
      labelPaddingParam: PaddingParam[T])
```
Apply an Optimizer who could apply padding to the Samples with a padding strategy.  
`model`: model will be optimizied.  
`sampleRDD`: training Samples.  
`criterion`: loss function.  
`batchSize`: mini batch size.  
`featurePaddingParam`: feature padding strategy.  
`labelPaddingParam`: label padding strategy.



```scala
def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      miniBatch: MiniBatch[T])
```
Apply an optimizer with User-Defined `MiniBatch`.  
`model`: model will be optimizied.  
`sampleRDD`: training Samples.  
`criterion`: loss function.  
`batchSize`: mini batch size.  
`miniBatch`: An User-Defined MiniBatch to construct a mini batch.

***Validation***

* **loss function**

In supervised machine learning, loss function compares the output of
the model with the ground truth(the labels of the training data). It
outputs a loss value to measure how good the model is(the lower the
better). It also provides a gradient to indicate how to tune the model.

In BigDL, all loss functions are subclass of Criterion.

* **batch size**

Training is an iterative process. In each iteration, only a batch of data
is used for training the model. You need to specify the batch size. Please note, 
the batch size should be divisible by the total cores number.

Here's an example of how to train a Linear classification model

**scala**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor

// Define the model
val model = Linear[Float](2, 1)
model.bias.zero()

// Generate 2D dummy data, y = 0.1 * x[1] + 0.3 * x[2]
val samples = Seq(
  Sample[Float](Tensor[Float](T(5f, 5f)), Tensor[Float](T(2.0f))),
  Sample[Float](Tensor[Float](T(-5f, -5f)), Tensor[Float](T(-2.0f))),
  Sample[Float](Tensor[Float](T(-2f, 5f)), Tensor[Float](T(1.3f))),
  Sample[Float](Tensor[Float](T(-5f, 2f)), Tensor[Float](T(0.1f))),
  Sample[Float](Tensor[Float](T(5f, -2f)), Tensor[Float](T(-0.1f))),
  Sample[Float](Tensor[Float](T(2f, -5f)), Tensor[Float](T(-1.3f)))
)
val trainData = sc.parallelize(samples, 1)

// Define the model
val optimizer = Optimizer[Float](model, trainData, MSECriterion[Float](), 4)
Engine.init
optimizer.optimize()
println(model.weight)
```

The weight of linear is init randomly. But the output should be like
```
scala> println(model.weight)
0.09316949      0.2887804
[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x2]
```

**python**
```
from bigdl.nn.layer import Linear
from bigdl.util.common import *
from bigdl.nn.criterion import MSECriterion
from bigdl.optim.optimizer import Optimizer, MaxIteration
import numpy as np

model = Linear(2, 1)
samples = [
  Sample.from_ndarray(np.array([5, 5]), np.array([2.0])),
  Sample.from_ndarray(np.array([-5, -5]), np.array([-2.0])),
  Sample.from_ndarray(np.array([-2, 5]), np.array([1.3])),
  Sample.from_ndarray(np.array([-5, 2]), np.array([0.1])),
  Sample.from_ndarray(np.array([5, -2]), np.array([-0.1])),
  Sample.from_ndarray(np.array([2, -5]), np.array([-1.3]))
]
train_data = sc.parallelize(samples, 1)
init_engine()
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4)
optimizer.optimize()
model.get_weights()[0]
```

The output should be like
```
array([[ 0.11578175,  0.28315681]], dtype=float32)
```

You can see the model is trained.

### Define when to end the training
You need define when to end the training. It can be several iterations, or how many round
data you want to process, a.k.a epoch.

**scala**
```scala
// The default endWhen in scala is 100 iterations
optimizer.setEndWhen(Trigger.maxEpoch(10))  // Change to 10 epoch
```

**python**
```
# Python need to define in the constructor
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4)
```

### Change the optimization algorithm
Gradient based optimization algorithms are the most popular algorithms to train the neural
network model. The most famous one is SGD. SGD has many variants, adagrad, adam, etc.

**scala**
```scala
// The default is SGD
optimizer.setOptimMethod(new Adam())  // Change to adam
```

**python**
```
# Python need to define the optimization algorithm in the constructor
optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(100), 4, optim_method = Adam())
```

### Validate your model in training
Sometimes, people want to evaluate the model with a seperated dataset. When model
performs well on train dataset, but bad on validation dataset, we call the model is overfit or
weak generalization. People may want to evaluate the model every serveral iterations or 
epochs. BigDL can easily do this by

**scala**
```scala
optimizer.setValidation(trigger, testData, validationMethod, batchSize)
```

```python
optimizer.set_validation(batch_size, val_rdd, trigger, validationMethod)
```

For validation, you need to provide

* trigger: how often to do validation, maybe each several iterations or epochs
* test data: the seperate dataset for test
* validation method: how to evaluate the model, maybe top1 accuracy, etc.
* batch size: how many data evaluate in one time

### Visualize training process
See [Visualization with TensorBoard](https://bigdl-project.github.io/UserGuide/visualization-with-tensorboard/)
