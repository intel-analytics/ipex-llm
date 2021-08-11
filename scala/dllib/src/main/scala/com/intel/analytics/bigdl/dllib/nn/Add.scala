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

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * adds a bias term to input data ;
 *
 * @param inputSize size of input data
 */
@SerialVersionUID(4268487849759172896L)
class Add[T: ClassTag](val inputSize: Int
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  val bias = Tensor[T](inputSize)

  val ones : Tensor[T] = Tensor[T]()

  val gradBias : Tensor[T] = Tensor[T](inputSize)

  {
    val stdv = 1 / math.sqrt(bias.size(1))
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(biasInitMethod = bInit)
  }

  override def reset(): Unit = {
    biasInitMethod.init(bias, VariableFormat.ONE_D)
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    if (input.isSameSizeAs(bias)) {
      output.add(bias)
    } else {
      val batchSize = input.size(1)
      ones.resize(batchSize).fill(ev.one)
      val biasLocal = bias.view(bias.size.product)
      val outputLocal = output.view(batchSize, output.size.product/batchSize)
      outputLocal.addr(ev.fromType[Int](1), ones, biasLocal)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    gradInput.copy(gradOutput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (scaleB != 0) {
      if (gradBias.size(1) == 1) {
        gradBias(1) = gradBias(1).add(ev.times(ev.fromType[Double](scaleB), gradOutput.sum()))
      } else {
        if (input.isSameSizeAs(bias)) {
          gradBias.add(ev.fromType[Double](scaleB), gradOutput)
        } else {
          val gradOutputLocal = gradOutput.view(input.size(1),
            gradOutput.size.product/input.size(1))
          gradBias.view(gradBias.size().product).addmv(ev.fromType[Double](scaleB),
            gradOutputLocal.t(), ones)
        }
      }
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    ones.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.bias), Array(this.gradBias))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("bias" -> bias, "gradBias" -> gradBias))
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Add[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Add[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        bias == that.bias &&
        gradBias == that.gradBias &&
        inputSize == that.inputSize
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), bias, gradBias, inputSize)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Add {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int)(implicit ev: TensorNumeric[T]) : Add[T] = {
    new Add[T](inputSize)
  }
}
