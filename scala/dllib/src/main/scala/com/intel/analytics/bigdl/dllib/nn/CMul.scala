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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * This layer has a weight tensor with given size. The weight will be multiplied element wise to
 * the input tensor. If the element number of the weight tensor match the input tensor, a simply
 * element wise multiply will be done. Or the bias will be expanded to the same size of the input.
 * The expand means repeat on unmatched singleton dimension(if some unmatched dimension isn't
 * singleton dimension, it will report an error). If the input is a batch, a singleton dimension
 * will be add to the first dimension before the expand.
 *
 * @param size the size of the bias
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(8888147326550637025L)
class CMul[@specialized(Float, Double) T: ClassTag](
  val size: Array[Int])(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val weight: Tensor[T] = Tensor[T](size)
  val gradWeight : Tensor[T] = Tensor[T](size)

  private val _sum = Tensor[T]()
  private val _repeat = Tensor[T]()

  reset()

  override def reset(): Unit = {
    val stdv = 1.0/math.sqrt(weight.nElement())
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    if (input.nElement() == weight.nElement()) {
      output.cmul(weight)
    } else {
      val expand = if (weight.dim() == input.dim()) {
        weight.view(weight.size())
      } else {
        weight.view(Array(1) ++ weight.size())
      }
      val pivotDim = Utils.getOnlyDimGtOne(expand.size())
      if (pivotDim > 0) {
        mulOneDimWeight(pivotDim, expand, output)
      } else {
        expand.expandAs(output)
        output.cmul(expand)
      }
    }
    output
  }

  private def mulOneDimWeight(dim: Int, expand: Tensor[T], output: Tensor[T]): Unit = {
    val (innerNum, outerNum) = Utils.getInnerOuterNum(dim, output)
    val weightData = expand.storage().array()
    val weightOffset = expand.storageOffset() - 1
    var outer = 0
    var offset = output.storageOffset() - 1
    while (outer < outerNum) {
      var k = 0
      while (k < expand.nElement()) {
        ev.scal(innerNum, weightData(k + weightOffset), output.storage().array(), offset, 1)
        offset += innerNum
        k += 1
      }
      outer += 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    if (weight.nElement() == gradOutput.nElement()) {
      gradInput.addcmul(ev.fromType[Int](1), weight, gradOutput)
    } else {
      val expand = if (weight.dim() == gradOutput.dim()) {
        weight.view(weight.size())
      } else {
        weight.view(Array(1) ++ weight.size())
      }
      val pivotDim = Utils.getOnlyDimGtOne(expand.size())
      if (pivotDim > 0) {
        gradInput.copy(gradOutput)
        mulOneDimWeight(pivotDim, expand, gradInput)
      } else {
        expand.expandAs(gradOutput)
        gradInput.cmul(expand, gradOutput)
      }
    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {

    if (weight.nElement() == gradOutput.nElement()) {
      gradWeight.addcmul(ev.fromType[Double](scale), input, gradOutput)
    } else {
      if (weight.dim() == input.dim()) {
        _repeat.resizeAs(input).cmul(input, gradOutput)
        var sumFrom = _repeat
        var sumInto = _sum
        var i = 1
        while (i <= weight.dim()) {
          if (weight.size(i) != input.size(i)) {
            sumInto.sum(sumFrom, i)
            sumInto = sumFrom
            sumFrom = if (sumFrom == _repeat) _sum else _repeat
          }
          i += 1
        }
        gradWeight.add(ev.fromType[Double](scale), sumFrom)
      } else {
        _repeat.resizeAs(input).cmul(input, gradOutput)
        _sum.sum(_repeat, 1)
        gradWeight.add(ev.fromType[Double](scale), _sum)
      }

    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
  }

  override def clearState(): this.type = {
    super.clearState()
    _repeat.set()
    _sum.set()
    this
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[CMul[T]]) {
      return false
    }
    val other = obj.asInstanceOf[CMul[T]]
    if (this.eq(other)) {
      return true
    }

    size == other.size &&
      gradWeight == other.gradWeight &&
      weight == other.weight
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + size.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + weight.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.CMul(${java.util.Arrays.toString(size)})"
  }
}

object CMul {
  def apply[@specialized(Float, Double) T: ClassTag](
      size: Array[Int])(implicit ev: TensorNumeric[T]) : CMul[T] = {
    new CMul[T](size)
  }
}
