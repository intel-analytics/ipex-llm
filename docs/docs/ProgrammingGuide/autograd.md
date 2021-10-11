
## Overview

`autograd` provides automatic differentiation for math operations, so that you can easily build your own *custom loss and layer* (in both Python and Scala), as illustracted below. (See more examples [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/autograd)). Conceptually we use reverse mode together with the chain rule for automatic differentiation. `Variable` is used to record the linkage of the operation history, which would generated a static directed acyclic graph for the backward execution. Within the execution graph, leaves are the input `variables` and roots are the output `variables`.

### CustomLoss
1.Define a custom function using `autograd`

```
from zoo.pipeline.api.autograd import *

def mean_absolute_error(y_true, y_pred):
   return mean(abs(y_true - y_pred), axis=1)
```


2. Use `CustomLoss` in `compile` method.
```
# You can pass the loss function directly into `loss`
model.compile(optimizer = SGD(), loss = mean_absolute_error)
model.fit(x = ..., y = ...)
```

4. Use `CustomLoss` in `nnframe` pipeline.
```
# 1) Create a CustomLoss object from function.
loss = CustomLoss(mean_absolute_error, y_pred_shape=[2], y_true_shape=[2])
# 2) Passing the CustomLoss object to NNClassifier.
classifier = NNClassifier(lrModel, loss, SeqToTensor([1000]))
```

5. Use `forward` and `backward` to evaluate a `CustomLoss` for debugging.

```
# y_pred_shape=[2] is a shape without batch
loss = CustomLoss(mean_absolute_error, y_pred_shape=[2], y_true_shape=[2])
error = loss.forward(y_true=np.random.uniform(0, 1, shape[3, 2]), y_pred=np.random.uniform(0, 1, shape[3, 2]))
grad = loss.backward(y_true=np.random.uniform(0, 1, shape[3, 2]), y_pred=np.random.uniform(0, 1, shape[3, 2]))
```


### Lambda layer
1.Define custom function using `autograd`

```
from zoo.pipeline.api.autograd import *
def add_one_func(x):
   return x + 1.0
```

2.Define model using Keras-style API and *custom `Lambda` layer*
```
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
model = Sequential().add(Dense(1, input_shape=(2,))) \
                   .add(Lambda(function=add_one_func))
# Evaluation for debug purpose.
model.forward(np.random.uniform(0, 1, shape[3, 2])) # 3 is the batch size

```

### Construct variable computation without `Lambda` layer

- The returning type for each operation is a `Variable`, so you can connect those `Variable` together freely without using `Lambda`. i.e `Dense[Float](3).from(input2)` or `input1 + input2`
- Shape inference is supported as well, which means you can check the output shape of a `Variable` by calling `get_output_shape()`

Python
```
import zoo.pipeline.api.autograd as auto
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *

input = Input(shape=[2, 20]) # create a variable
time = TimeDistributed(layer=Dense(30))(input) # time is a variable
t1 = time.index_select(1, 0) # t1 is a variable
t2 = time.index_select(1, 1)
diff = auto.abs(t1 - t2)
assert diff.get_output_shape() == (None, 30)
assert diff.get_input_shape() == (None, 30)
model = Model(input, diff)
data = np.random.uniform(0, 1, [10, 2, 20])
output = model.forward(data)
```

Scala
- In respect of backward compatibility, the scala API is slightly different with the python API.
- `layer.inputs(node)` would return a node(backward compatibility).
- `layer.from(variable)` would return a variable.(You may want to use this style as it can support autograd.)

```
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.layers._

val input1 = Variable[Float](inputShape = Shape(3))
val input2 = Variable[Float](inputShape = Shape(3))
val diff = AutoGrad.abs(input1 - Dense[Float](3).from(input2))
val model = Model[Float](input = Array(input1, input2), output = diff)
val inputValue = Tensor[Float](1, 3).randn()
// In scala, we use Table for multiple inputs. `T` is a short-cut for creating a Table.
val out = model.forward(T(inputValue, inputValue)).toTensor[Float]
```


### Define a model using trainable Parameter
Build a `Linear` Model (Wx + b) by using trainable `Parameter` which is equivalent to use `Dense` layer.
* Scala
```
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
val input = Variable[Float](Shape(3))
val w = Parameter[Float](Shape(2, 3)) // outputSize * inputSize
val bias = Parameter[Float](Shape(2))
val cDense = AutoGrad.mm(input, w, axes = List(1, 1)) + bias
val model = Model[Float](input = input, output = cDense)

```

* Python

```
from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.models import *
input = Variable((3,))
w = Parameter((2, 3)) # outputSize * inputSize
bias = Parameter((2,))
cDense = mm(input, w, axes = (1, 1)) + bias
model = Model(input = input, output = cDense)

```




