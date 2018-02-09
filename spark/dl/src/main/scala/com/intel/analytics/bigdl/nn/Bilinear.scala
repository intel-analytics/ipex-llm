/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Initializable}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * a bilinear transformation with sparse inputs,
 * The input tensor given in forward(input) is a table containing both inputs x_1 and x_2,
 * which are tensors of size N x inputDimension1 and N x inputDimension2, respectively.
 *
 * @param inputSize1   dimension of input x_1
 * @param inputSize2   dimension of input x_2
 * @param outputSize   output dimension
 * @param biasRes  The layer can be trained without biases by setting bias = false. otherwise true
 * @param wRegularizer : instance of [[Regularizer]]
 *                     (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer : instance of [[Regularizer]]
 *                     applied to the bias.
 */

@SerialVersionUID(-4838965135083645415L)
class Bilinear[T: ClassTag](
 val inputSize1: Int,
 val inputSize2: Int,
 val outputSize: Int,
 val biasRes: Boolean = true,
 var wRegularizer: Regularizer[T] = null,
 var bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] with Initializable {

  require((inputSize1 > 0) && (inputSize2 > 0) && (outputSize > 0),
    s"Bilinear: inputSize1 and inputSize2 and outputSize should be positive integer numbers," +
      s"but got inputSize1 $inputSize1, inputSize2 $inputSize2, outputSize $outputSize")

  val weight = Tensor[T](outputSize, inputSize1, inputSize2)
  val bias: Tensor[T] = if (biasRes) Tensor[T](outputSize) else null

  var buff1: Tensor[T] = Tensor[T]()
  var buff2: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] = Tensor[T](outputSize, inputSize1, inputSize2)
  val gradBias: Tensor[T] = Tensor[T](outputSize)

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    var wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    var bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.Default)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }

  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 2,
      s"Bilinear: input should be a table containing two data Tensors," +
        s"but got input.length ${input.length()}")
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    require(res1.nDimension() == 2 && res2.nDimension() == 2 && res1.size(1) == res2.size(1),
      "Bilinear: input Tensors should be two-dimensional and" +
        " have the same number of rows, " +
        s"res1[ ${res1.nDimension()}, ${res1.size(1)}]," +
        s" res2[ ${res2.nDimension()}, ${res2.size(1)} ]")
    require(res1.size(2) == weight.size(2) && res2.size(2) == weight.size(3),
      "Bilinear: dimensionality of first input and second input is erroneous," +
        s" first ${res1.size(2)}, " +
        s"second ${res2.size(2)}")

    // set up buffer
    buff2.resizeAs(res2)

    // compute output scores
    output.resize(res1.size(1), weight.size(1))
    var k = 1
    while (k < (weight.size(1) + 1)) {
      buff2.zero()
      buff2.addmm(res1, weight(k))
      buff2.cmul(res2)
      output.narrow(2, k, 1).sum(buff2, 2)
      k += 1
    }
    if (bias != null) {
      output.add(bias.reshape(Array(1, bias.nElement())).expand(output.size()))
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    require(res1.size(1) == gradOutput.size(1),
      s"Bilinear: number of rows in gradOutput does not match input, " +
        s"got input rows ${res1.size(1)} and gradOutput rows ${gradOutput.size(1)}")
    require(gradOutput.size(2) == weight.size(1),
      s"Bilinear: number of columns in gradOutput does not output size of layer, " +
        s"got gradOutput columns ${gradOutput.size(2)} and output columns ${weight.size(1)}")

    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T]())
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T]())

    val gradInput1 = gradInput[Tensor[T]](1)
    val gradInput2 = gradInput[Tensor[T]](2)

    // compute d output / d input:
    gradInput1.resizeAs(res1).zero()
    gradInput2.resizeAs(res2).zero()

    // do first slice of weight tensor (k = 1)
    gradInput1.addmm(res2, weight.select(1, 1).t())
    gradInput1.cmul(gradOutput.narrow(2, 1, 1).expand(
      Array(gradInput1.size(1), gradInput1.size(2))))

    gradInput2.addmm(ev.fromType(1), res1, weight.select(1, 1))
    gradInput2.cmul(gradOutput.narrow(2, 1, 1).expand(
      Array(gradInput2.size(1), gradInput2.size(2))))

    // do remaining slices of weight tensor
    if (weight.size(1) > 1) {
      buff1.resizeAs(res1)

      var k = 2
      while (k < (weight.size(1) + 1)) {
        buff1.zero()
        buff2.zero()

        buff1.addmm(res2, weight.select(1, k).t())
        buff1.cmul(gradOutput.narrow(2, k, 1).expand(
          Array(gradInput1.size(1), gradInput1.size(2))))
        gradInput1.add(buff1)

        buff2.addmm(input(1), weight.select(1, k))
        buff2.cmul(gradOutput.narrow(2, k, 1).expand(
          Array(gradInput2.size(1), gradInput2.size(2))))
        gradInput2.add(buff2)
        k += 1
      }
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Tensor[T]): Unit = {
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    // make sure we have buffer
    if (null == buff1) buff1 = Tensor[T]()
    buff1.resizeAs(res1)

    // accumulate parameter gradients:
    if (scaleW !=0 ) {
      var k = 1
      while (k < (weight.size(1) + 1)) {
        buff1.zero()
        buff1.cmul(res1, gradOutput.narrow(2, k, 1).expandAs(res1))
        gradWeight.select(1, k).addmm(ev.fromType[Double](scaleW), buff1.t(), input(2))
        k += 1
      }
    }

    if(null != bias && scaleB != 0) gradBias.add(ev.fromType[Double](scaleB), gradOutput.sum(1))

    if (wRegularizer != null && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (bRegularizer != null && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def clearState(): this.type = {
    super.clearState()
    buff1.set()
    buff2.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize1, $inputSize2, $outputSize, $biasRes)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Bilinear[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Bilinear[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        weight == that.weight &&
        bias == that.bias &&
        gradWeight == that.gradWeight &&
        gradBias == that.gradBias &&
        inputSize1 == that.inputSize1 &&
        inputSize2 == that.inputSize2 &&
        outputSize == that.outputSize &&
        biasRes == that.biasRes
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()

    val state = Seq(super.hashCode(), weight, bias, gradWeight, gradBias,
      inputSize1, inputSize2, outputSize, biasRes)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Bilinear {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize1: Int,
    inputSize2: Int,
    outputSize: Int,
    biasRes: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
   )(implicit ev: TensorNumeric[T]): Bilinear[T] = {
    new Bilinear[T](inputSize1, inputSize2, outputSize, biasRes,
      wRegularizer, bRegularizer)
  }
}
