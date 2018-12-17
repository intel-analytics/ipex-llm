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

import com.intel.analytics.bigdl.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a transfer module that applies LeakyReLU, which parameter
 * negval sets the slope of the negative part:
 * LeakyReLU is defined as:
 *  f(x) = max(0, x) + negval * min(0, x)
 *
 * @param negval sets the slope of the negative partl
 * @param inplace if it is true, doing the operation in-place without
 *                using extra state memory
 */

@SerialVersionUID(- 6870619109313859155L)
class LeakyReLU[T: ClassTag](
  private val negval: Double = 0.01,
  var inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  import LeakyReLU._

  if (negval < 0) {
    inplace = false
  }


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous(), "input should be contiguous")
    if (inplace) output = input
    input.getType() match {
      case FloatType => updateOutputFloat(input.toTensor[Float], output.toTensor[Float],
        negval.toFloat, inplace)
      case DoubleType => updateOutputDouble(input.toTensor[Double], output.toTensor[Double],
        negval, inplace)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "input should have the same size with gradOutput" +
        s"input size ${input.dim()} gradOutput size ${gradOutput.dim()}")
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    if (inplace) gradInput = gradOutput
    input.getType() match {
      case FloatType => updateGradInputFloat(input.toTensor[Float], gradOutput.toTensor[Float],
        gradInput.toTensor[Float], negval.toFloat, inplace)
      case DoubleType => updateGradInputDouble(input.toTensor[Double], gradOutput.toTensor[Double],
        gradInput.toTensor[Double], negval, inplace)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }
}

object LeakyReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
      negval: Double = 0.01,
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : LeakyReLU[T] = {
    new LeakyReLU[T](negval, inplace)
  }

  protected def updateOutputFloat(
        input: Tensor[Float],
        output: Tensor[Float],
        negVal: Float,
        inplace: Boolean): Unit = {
    if (inplace) {
      var i = input.storageOffset() - 1
      val array = input.storage().array()
      val end = input.nElement() + input.storageOffset() - 1
      while (i < end) {
        if (array(i) < 0) {
          array(i) *= negVal
        }
        i += 1
      }
    } else {
      output.resizeAs(input)
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val outputOffset = output.storageOffset() - 1
      val outputArray = output.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) < 0) {
          outputArray(i + outputOffset) = inputArray(i + inputOffset) * negVal
        } else {
          outputArray(i + outputOffset) = inputArray(i + inputOffset)
        }
        i += 1
      }
    }
  }

  protected def updateOutputDouble(
        input: Tensor[Double],
        output: Tensor[Double],
        negVal: Double,
        inplace: Boolean): Unit = {
    if (inplace) {
      var i = input.storageOffset() - 1
      val array = input.storage().array()
      val end = input.nElement() + input.storageOffset() - 1
      while (i < end) {
        if (array(i) < 0) {
          array(i) *= negVal
        }
        i += 1
      }
    } else {
      output.resizeAs(input)
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val outputOffset = output.storageOffset() - 1
      val outputArray = output.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) < 0) {
          outputArray(i + outputOffset) = inputArray(i + inputOffset) * negVal
        } else {
          outputArray(i + outputOffset) = inputArray(i + inputOffset)
        }
        i += 1
      }
    }
  }

  protected def updateGradInputFloat(
        input: Tensor[Float],
        gradOutput: Tensor[Float],
        gradInput: Tensor[Float],
        negVal: Float,
        inplace: Boolean): Unit = {
    if (inplace) {
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1
      val gradInputArray = gradInput.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) > 0) {
          gradInputArray(i + gradInputOffset) *= negVal
        }
        i += 1
      }
    } else {
      gradInput.resizeAs(input)
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val gradOutputOffset = gradOutput.storageOffset() - 1
      val gradOutputArray = gradOutput.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1
      val gradInputArray = gradInput.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) < 0) {
          gradInputArray(i + gradInputOffset) = gradOutputArray(i + gradOutputOffset) * negVal
        } else {
          gradInputArray(i + gradInputOffset) = gradOutputArray(i + gradOutputOffset)
        }
        i += 1
      }
    }
  }

  protected def updateGradInputDouble(
        input: Tensor[Double],
        gradOutput: Tensor[Double],
        gradInput: Tensor[Double],
        negVal: Double,
        inplace: Boolean): Unit = {
    if (inplace) {
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1
      val gradInputArray = gradInput.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) > 0) {
          gradInputArray(i + gradInputOffset) *= negVal
        }
        i += 1
      }
    } else {
      gradInput.resizeAs(input)
      var i = 0
      val inputOffset = input.storageOffset() - 1
      val inputArray = input.storage().array()
      val gradOutputOffset = gradOutput.storageOffset() - 1
      val gradOutputArray = gradOutput.storage().array()
      val gradInputOffset = gradInput.storageOffset() - 1
      val gradInputArray = gradInput.storage().array()
      val end = input.nElement()
      while (i < end) {
        if (inputArray(i + inputOffset) < 0) {
          gradInputArray(i + gradInputOffset) = gradOutputArray(i + gradOutputOffset) * negVal
        } else {
          gradInputArray(i + gradInputOffset) = gradOutputArray(i + gradOutputOffset)
        }
        i += 1
      }
    }
  }
}
