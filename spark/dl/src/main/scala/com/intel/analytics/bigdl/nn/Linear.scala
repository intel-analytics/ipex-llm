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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.reflect.ClassTag
import RandomGenerator._
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.Regularizer

/**
 * The [[Linear]] module applies a linear transformation to the input data,
 * i.e. `y = Wx + b`. The input given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
 * have the size of `inputSize`. If it is a matrix, then each row is assumed to be
 * an input sample of given batch (the number of rows means the batch size and
 * the number of columns should be equal to the `inputSize`).
 *
 * @param inputSize the size the each input sample
 * @param outputSize the size of the module output of each sample
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */
@SerialVersionUID( 359656776803598943L)
class Linear[T: ClassTag](
  inputSize: Int,
  outputSize: Int,
  withBias: Boolean = true,
  wRegularizer: Regularizer[T] = null,
  bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val weight: Tensor[T] = Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] = if (withBias) Tensor[T](outputSize) else null
  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] = Tensor[T]()
  val gradBias: Tensor[T] = if (withBias) Tensor[T]() else null

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    reset(wInit, bInit)
  }

  override protected def resetInternal(): Unit = {
    weightInitMethod.init(weight, VariableFormat.OUT_IN)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
      if (output.nElement() != nElement) {
        output.zero()
      }

      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.fromType[Int](1))
      }

      output.addmm(ev.fromType[Int](0), output, ev.fromType[Int](1), input, weight.t)
      if (withBias) output.addr(ev.fromType[Int](1), addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    val value = ev.fromType[Double](scale)
    if (input.dim() == 1) {
      gradWeight.addr(value, gradOutput, input)
      if (withBias) {
        gradBias.add(value, gradOutput)
      }
    }
    else if (input.dim() == 2) {
      gradWeight.addmm(value, gradOutput.t, input)
      if (withBias) gradBias.addmv(value, gradOutput.t, addBuffer)
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight)
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias)
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.add(ev.negative(learningRate), gradWeight)
    if (withBias) bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.resize(outputSize, inputSize)
    gradWeight.zero()
    if (withBias) {
      gradBias.resize(outputSize)
      gradBias.zero()
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def getParametersTable(): Table = {
    if (null == bias) {
      T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
    } else {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    }
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize -> $outputSize)"
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputSize: Int,
      outputSize: Int,
      withBias: Boolean = true,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize,
      withBias, wRegularizer, bRegularizer)
  }
}
