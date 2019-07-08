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
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T, Table}

import scala.reflect.ClassTag

/**
 * This layer has a bias tensor with given size. The bias will be added element wise to the input
 * tensor. If the element number of the bias tensor match the input tensor, a simply element wise
 * will be done. Or the bias will be expanded to the same size of the input. The expand means
 * repeat on unmatched singleton dimension(if some unmatched dimension isn't singleton dimension,
 * it will report an error). If the input is a batch, a singleton dimension will be add to the first
 * dimension before the expand.
 *
 * @param size the size of the bias
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(3917196591309935383L)
class CAdd[T: ClassTag](
  val size: Array[Int],
  var bRegularizer: Regularizer[T] = null)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  val bias: Tensor[T] = Tensor[T](size)
  val gradBias : Tensor[T] = Tensor[T](size)

  {
    val stdv = 1.0/math.sqrt(bias.nElement())
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(biasInitMethod = bInit)
  }

  override def reset(): Unit = {
    biasInitMethod.init(bias, VariableFormat.ONE_D)
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    if (input.nElement() == bias.nElement()) {
      output.add(bias)
    } else {
      val expand = if (bias.dim() == input.dim()) {
        bias.view(bias.size())
      } else {
        bias.view(Array(1) ++ bias.size())
      }
      val pivotDim = Utils.getOnlyDimGtOne(bias.size())
      if (pivotDim > 1) {
        addOneDimBias(pivotDim, expand, output)
      } else {
        expand.expandAs(output)
        output.add(expand)
      }
    }
    output
  }

  private def addOneDimBias(pivotDim: Int, expand: Tensor[T], output: Tensor[T]): Unit = {
    val (innerNum, outerNum) = Utils.getInnerOuterNum(pivotDim, output)
    val biasData = expand.storage().array()
    val biasOffset = expand.storageOffset() - 1
    var outer = 0
    var offset = output.storageOffset() - 1
    var k = 0
    while (outer < outerNum) {
      k = 0
      while (k < expand.nElement()) {
        ev.add(innerNum, output.storage().array(), offset, biasData(k + biasOffset), 1)
        offset += innerNum
        k += 1
      }
      outer += 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (scaleB == 0) {
      return
    }
    if (bias.nElement() == gradOutput.nElement()) {
      gradBias.add(ev.fromType[Double](scaleB), gradOutput)
    } else {
      val expand = if (bias.dim() == gradOutput.dim()) {
        gradBias.view(gradBias.size())
      } else {
        gradBias.view(Array(1) ++ gradBias.size())
      }
      val pivotDim = Utils.getOnlyDimGtOne(bias.size())
      if (pivotDim > 1) {
        val (innerNum, outerNum) = Utils.getInnerOuterNum(pivotDim, output)
        val biasData = expand.storage().array()
        var outer = 0
        var offset = output.storageOffset() - 1
        var k = 0
        val gradOutputData = gradOutput.storage().array()
        while (outer < outerNum) {
          k = 0
          while (k < expand.nElement()) {
            biasData(k) = ev.plus(ev.times(ev.sum(innerNum, gradOutputData, offset, 1),
              ev.fromType[Double](scaleB)), biasData(k))
            offset += innerNum
            k += 1
          }
          outer += 1
        }
      } else {
        expand.expandAs(gradOutput)
        expand.add(ev.fromType[Double](scaleB), gradOutput)
      }
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.bias), Array(this.gradBias))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("bias" -> bias, "gradBias" -> gradBias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[CAdd[T]]) {
      return false
    }
    val other = obj.asInstanceOf[CAdd[T]]
    if (this.eq(other)) {
      return true
    }

    size == other.size &&
      gradBias == other.gradBias &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + size.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}(${java.util.Arrays.toString(size)})"
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object CAdd {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: Array[Int],
    bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : CAdd[T] = {
    new CAdd[T](size, bRegularizer)
  }
}
