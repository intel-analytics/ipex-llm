BigDL supports two different model definition styles: Sequential API and Functional API.

In Functional API, the model is described as a graph. It is more convenient than Sequential API
when define some complex model.

---
## **Define a simple model**
Suppose we want to define a model with three layers
```
Linear -> Sigmoid -> SoftMax
```

You can write code like this

**Scala:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val inputSize = 3
val outputSize = 5
val input = Tensor(inputSize).rand()

val linear = Linear(inputSize, outputSize).inputs()
val sigmoid = Sigmoid().inputs(linear)
val softmax = SoftMax().inputs(sigmoid)
val model = Graph(Array(linear), Array(softmax))

val output = model.forward(input)
    
print(output)
```
**Python:**
```python
import numpy as np
from bigdl.nn.layer import *

input_size = 3
output_size = 5
input = np.random.random(input_size)

linear = Linear(3, 5)()
sigmoid = Sigmoid()(linear)
softmax = SoftMax()(sigmoid)
model = Model([linear], [softmax])

model.forward(input)
```

An easy way to understand the Functional API is to think of each layer in the model as a directed
edge connecting its input and output

In the above code, first we create an input node named as linear by using
the Linear layer, then connect it to the sigmoid node with a Sigmoid
layer, then connect the sigmoid node to the softmax node with a SoftMax layer.

After defined the graph, we create the model by passing in the input nodes
and output nodes.

---
## **Define a model with branches**
Suppose we want to define a model like this
```
Linear -> ReLU --> Linear -> ReLU
               |-> Linear -> ReLU
```
The model has two outputs from two branches. The inputs of the branches are both the
output from the first ReLU.

You can define the model like this

**Scala:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val inputSize = 3
val midLayerSize = 10
val outputSize = 5
val input = Tensor(inputSize).rand()

val linear1 = Linear(inputSize, midLayerSize).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(midLayerSize, outputSize).inputs(relu1)
val relu2 = ReLU().inputs(linear2)
val linear3 = Linear(midLayerSize, outputSize).inputs(relu1)
val relu3 = ReLU().inputs(linear3)
val model = Graph(Array(linear1), Array(relu2, relu3))

val output = model.forward(input)

print(output)
```
**Python:**
```python
import numpy as np
from bigdl.nn.layer import *

input_size = 3
mid_layer_size = 10
output_size = 5
input = np.random.random(input_size)

linear1 = Linear(input_size, mid_layer_size)()
relu1 = ReLU()(linear1)
linear2 = Linear(mid_layer_size, output_size)(relu1)
relu2 = ReLU()(linear2)
linear3 = Linear(mid_layer_size, output_size)(relu1)
relu3 = ReLU()(linear3)
model = Model([linear1], [relu2, relu3])

model.forward(input)
```
In the above node, linear2 and linear3 are both from relu1 with separated
Linear layers, which construct the branch structure. When we create the model,
the outputs parameter contains relu2 and relu3 as the model has two outputs.

---
## **Define a model with merged branch**
Suppose we want to define a model like this
```
Linear -> ReLU --> Linear -> ReLU ----> Add
               |-> Linear -> ReLU --|
```
In the model, the outputs of the two branches are merged by an add operation.

You can define the model like this

**Scala:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val inputSize = 3
val midLayerSize = 10
val outputSize = 5
val input = Tensor(inputSize).rand()

val linear1 = Linear(inputSize, midLayerSize).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(midLayerSize, outputSize).inputs(relu1)
val relu2 = ReLU().inputs(linear2)
val linear3 = Linear(midLayerSize, outputSize).inputs(relu1)
val relu3 = ReLU().inputs(linear3)
val add = CAddTable().inputs(relu2, relu3)
val model = Graph(Array(linear1), Array(add))

val output = model.forward(input)

print(output)
```
**Python:**
```python
import numpy as np
from bigdl.nn.layer import *

input_size = 3
mid_layer_size = 10
output_size = 5
input = np.random.random(input_size)

linear1 = Linear(input_size, mid_layer_size)()
relu1 = ReLU()(linear1)
linear2 = Linear(mid_layer_size, output_size)(relu1)
relu2 = ReLU()(linear2)
linear3 = Linear(mid_layer_size, output_size)(relu1)
relu3 = ReLU()(linear3)
add = CAddTable()([relu2, relu3])
model = Model([linear1], [add])

model.forward(input)
```
In the above code, to merge the branch, we use the CAddTable, which takes two
input nodes, to generate one output node.

BigDL provides many merge layers. Please check Merge layers document page. They all
take a list of tensors as input and merge the tensors by some operation.

---
## **Define a model with multiple inputs**
We have already seen how to define branches in model and how to merge branches.
What if we have multiple input? Suppose we want to define a model like this
```
Linear -> ReLU ----> Add
Linear -> ReLU --|
```

You can define the model like this

**Scala:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val inputSize = 3
val midLayerSize = 10
val outputSize = 5
val input = T(
  Tensor(inputSize).rand(),
  Tensor(inputSize).rand()
)

val linear1 = Linear(inputSize, outputSize).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(inputSize, outputSize).inputs()
val relu2 = ReLU().inputs(linear2)
val add = CAddTable().inputs(relu1, relu2)
val model = Graph(Array(linear1, linear2), Array(add))

val output = model.forward(input)

print(output)
```
**Python:**
```python
import numpy as np
from bigdl.nn.layer import *

input_size = 3
mid_layer_size = 10
output_size = 5
linear1_input = np.random.random(input_size)
linear2_input = np.random.random(input_size)
input = [linear1_input, linear2_input]

linear1 = Linear(input_size, output_size)()
relu1 = ReLU()(linear1)
linear2 = Linear(input_size, output_size)()
relu2 = ReLU()(linear2)
add = CAddTable()([relu1, relu2])
model = Model([linear1, linear2], [add])

model.forward(input)
```
In the above code, we define two input nodes linear1 and linear2 and put them
into the first parameter when create the graph model.

## **Define a model with data pre-processing**
You can use a model as the data preprocessor for another model. In the training, the parameter of
the preprocessor won't be updated. Here's how you can use it:

**Scala:**
```scala
val preprocessor = Module.Load(...)
val trainable = Module.Load(...)
val model = Graph(preprocessor, trainable)
```

**Python:**
```python
preprocessor = Model.loadModel(...)
trainable = Model.loadModel(...)
model = Model(preprocessor, trainable)
```
