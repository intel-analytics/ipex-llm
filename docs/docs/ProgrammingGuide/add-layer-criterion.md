## How to add your own layer or criterion into BigDL
If you'd like to create a layer or criterion which has not been covered by BigDL library, you just
need to extend the AbstractModule or AbstractCriterion class in scala.

Here we show how to do it with some examples.

## Create a layer in scala
Say we want to create a layer which adds one to each element of the input tensor. We can write code
like this:
```scala
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

class AddOneLayer extends AbstractModule[Tensor[Float], Tensor[Float], Float]{

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    output.resizeAs(input).copy(input).add(1.0f)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    gradInput.resizeAs(gradOutput).copy(gradOutput)
    gradInput
  }
}
```
In the above code piece, we create a new Layer class by extending the AbstractModule. AbstractModule
has three generic type: **input data type**, **output data type** and **parameter type**. In this
example, the new layer takes a float tensor as input, and outputs a float tensor.

We override two methods, **updateOutput** and **updateGradInput**

* updateOutput

In the forward process, each layer invokes the updateOutput method to process the input data. The
**AddOneLayer** copy the input data to the output tensor, and then add one to each element.

You may notice that we don't change the input tensor. The input tensor may be used in serval layers,
so change it may cause incorrect result.

Each layer has an buffer named as **output**. The output is cached in that buffer to be used by the
succeed layers. The output buffer is inited as an empty tensor, so we need to resize it when we use.

* updateGradInput

In the backward process, each layer invokes the updateGradInput method to back propagate the
gradients. Note the direction is backward, so the layer takes a gradOutput and produce a gradInput.

For how the backward works, please check the [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule).
The **AddOneLayer** just back propagate the gradients identity to its prior layers. We don't modify
the gradOutput for the same reason in the updateOutput method.

## Create a layer with multiple inputs
When a layer has multiple input data(e.g. multiple tensors), we can use **Table** as input type of
the layer. **Table** is a nested type.

Say we want to create a layer, which takes two float tensors as input, and add them together as
output.

Here's the code example:
```scala
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class AddTwoTensorLayer extends AbstractModule[Table, Tensor[Float], Float]{

  override def updateOutput(input: Table): Tensor[Float] = {
    val firstTensor = input[Tensor[Float]](1)
    val secondTensor = input[Tensor[Float]](2)
    output.resizeAs(firstTensor).copy(firstTensor).add(secondTensor)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    val firstGrad = gradInput.getOrElse[Tensor[Float]](1, Tensor[Float]())
    val secondGrad = gradInput.getOrElse[Tensor[Float]](2, Tensor[Float]())
    firstGrad.resizeAs(gradOutput).copy(gradOutput)
    secondGrad.resizeAs(gradOutput).copy(gradOutput)
    gradInput(1) = firstGrad
    gradInput(2) = secondGrad
    gradInput
  }
}
```

When use table, we provide the index key(start from 1) and the type. Please note that as the input
type is Table, the gradInput buffer is also inited as an empty Table.

## Create a layer with multiple outputs
When a layer has multiple outputs, we just need to specify the output type as **Table**.

Say we want a layer to split a N-d tensor into many (N - 1)-d tensors in the first dimension. Here
is the code example:

```scala
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class UnpackLayer extends AbstractModule[Tensor[Float], Table, Float]{

  override def updateOutput(input: Tensor[Float]): Table = {
    require(input.nDimension() > 1)
    output = T()
    (1 to input.size(1)).foreach(i => {
      output(i) = Tensor[Float]().resizeAs(input.select(1, i)).copy(input.select(1, i))
    })
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Table): Tensor[Float] = {
    gradInput.resizeAs(input)
    (1 to input.size(1)).foreach(i => {
      gradInput.select(1, i).copy(gradOutput[Tensor[Float]](i))
    })
    gradInput
  }
}
```


## Create a layer with parameters
Some layers are trainable, which means they have some parameters and the parameters are changed in
the training process. Trainable layer need to override extra methods.

Parameters are all tensors and their numeric type is determined by the third generic type of the
AbstractModule.

Say we want to create a tensor which multiply a N x 3 matrix with a 3 x 2 parameter, and get a N x 2
output. It looks like our first example. Here is the initial code:
```scala
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class MatrixLayer extends AbstractModule[Tensor[Float], Tensor[Float], Float]{

  private val w = Tensor[Float](3, 2)

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    require(input.size(2) == 3)
    output.resize(input.size(1), 2).zero().addmm(input, w)
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    gradInput.resizeAs(input)
    gradInput.zero().addmm(gradOutput, w.t())
  }
}
```

However, the parameter w here is not trainable. We need override several methods. Here's the code
example
```scala
  private val g = Tensor[Float](3, 2)

  override def accGradParameters(input: Tensor[Float], gradOutput: Tensor[Float]): Unit = {
    g.addmm(gradOutput, input.t())
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(w), Array(g))
  }

  override def zeroGradParameters(): Unit = {
    g.zero()
  }

  override def reset(): Unit = {
      w.rand()
      g.zero()
  }
```
First we introduce a new tensor g, which has same size with parameter w. We call g gradParameter, as
it stores the gradients which will be used to update w.

* accGradParameters

In this method, we cacluate the gradient of w and accumulate it to gradParameter g

* parameters

Return parameters and gradient parameters of this layer. Default it return nothing. In this example,
we return w and g. Please note that some layer can have multiple parameters.

* zeroGradParameters

Reset gradParameters to zero.

* reset

ReInitialize the parameter and reset gradParameter to zero

**How to use gradient to update parameter**

You may notice that in the layer we don't define how to use g to update w. Layers just provide
gradient and parameters. How to use gradient to update parameter is handled by
[optimize methods](/APIGuide/Optimizers/Optim-Methods.md) and
[optimizer](/APIGuide/Optimizers/Optimizer.md).

## Create a criterion in scala
To create your own criterion, you need to extend the **AbstractCriterion** class. The criterion
take two inputs, and calculate some scalar 'distance' between them, and the gradient indicting how
to reduce the 'distance'.

Let's create a simple criterion to demostrate how it works. This criterion will calculate the sum of
abstract value of elementwise difference between the two input tensors. Here is the code:
```scala
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class AbsCriterion extends AbstractCriterion[Tensor[Float], Tensor[Float], Float]{

  override def updateOutput(input: Tensor[Float], target: Tensor[Float]): Float = {
    (input - target).abs().sum
  }

  override def updateGradInput(input: Tensor[Float], target: Tensor[Float]): Tensor[Float] = {
    gradInput.resizeAs(input).copy(input)
    gradInput.map(target, (a, b) => if (a > b) 1 else if (a == b) 0 else -1)
  }
}
```

AbstractCeiterion has three generic types: **first input type**, **second input type** and
**loss type**. In this example, the first input and the second input are both tensors. The loss
is float.

We need to override two method: **updateOutput** and **updateGradInput**. It's similiar to layer.

* updateOutput

Calculate the loss from the given two inputs. In this example, we first element-wise sub the two
tensor, then calculate the abstract value and then sum the result together.

* updateGradInput

Comput the gradient against the input tensor. In this example, we element-wise compare the two
tensors. If input > target, the gradient is 1, if input == target, the gradient is 0, else the
gradient is -1.
