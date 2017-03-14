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
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.Future
import scala.reflect.ClassTag

@SerialVersionUID(- 7842335603491194236L)
class SoftMax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  @transient
  private var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(1 <= input.nDimension() && input.nDimension() <= 4,
      "1D, 2D, 3D or 4D tensor expected")
    val (nFrame, stride) = if (input.nDimension() == 1) {
      (1, 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(3) * input.size(4))
    }
    if (results == null || results.length != nFrame * stride) {
      results = new Array[Future[Unit]](nFrame * stride)
    }
    output.resizeAs(input)
    SoftMax.updateOutput[T](input, output, results)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    SoftMax.updateGradInput[T](input, gradOutput, gradInput, output, results)
    gradInput
  }

  override def toString(): String = {
    s"nn.SoftMax"
  }
}

object SoftMax{

  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : SoftMax[T] = {
    new SoftMax[T]()
  }
  // Notice: SoftMin will call this function
  private[nn] def updateOutput[T: ClassTag](input: Tensor[T], output: Tensor[T],
    results: Array[Future[Unit]]) (implicit ev: TensorNumeric[T]): Tensor[T] = {

    val (nFrame, dim, stride) = if (input.nDimension() == 1) {
      (1, input.size(1), 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), input.size(2), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(1), input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(2), input.size(3) * input.size(4))
    }

    val outputArray = output.storage().array()
    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }

    var t = 0
    while (t < stride * nFrame) {
      val _t = t
      results(_t) = Engine.model.invoke(() => {
        val inputOffset = (_t / stride) * dim * stride + _t % stride
        val outputOffset = (_t / stride) * dim * stride + _t % stride

        var inputMax = ev.fromType[Float](Float.MinValue)

        var d = 0
        while (d < dim) {
          if (ev.isGreater(inputArray(d * stride + inputOffset), inputMax)) {
            inputMax = inputArray(d * stride + inputOffset)
          }
          d += 1
        }

        var sum = ev.fromType[Int](0)
        d = 0
        while (d < dim) {
          val z = ev.exp(ev.minus(inputArray(d * stride + inputOffset), inputMax))
          outputArray(d * stride + outputOffset) = z
          sum = ev.plus(sum, z)
          d += 1
        }

        d = 0
        while (d < dim) {
          outputArray(d * stride + outputOffset) =
            ev.times(outputArray(d * stride + outputOffset), ev.divide(ev.fromType[Int](1), sum))
          d += 1
        }
      })

      t += 1
    }
    Engine.model.sync(results)

    output
  }

  private[nn] def updateGradInput[T: ClassTag](input: Tensor[T], gradOutput: Tensor[T],
    gradInput: Tensor[T], output: Tensor[T],
    results: Array[Future[Unit]])(implicit ev: TensorNumeric[T]): Tensor[T] = {

    require(input.size().deep == gradOutput.size().deep,
      "input should have the same size with gradOutput")
    val (nFrame, dim, stride) = if (output.nDimension() == 1) {
      (1, output.size(1), 1)
    } else if (output.nDimension() == 2) {
      (output.size(1), output.size(2), 1)
    } else if (output.nDimension() == 3) {
      (1, output.size(1), output.size(2) * output.size(3))
    } else {
      (output.size(1), output.size(2), output.size(3) * output.size(4))
    }

    val gradInputArray = gradInput.storage().array()
    val outputArray = if (output.isContiguous()) {
      output.storage().array()
    } else {
      output.contiguous().storage().array()
    }
    val gradOutputArray = if (gradOutput.isContiguous()) {
      gradOutput.storage().array()
    } else {
      gradOutput.contiguous().storage().array()
    }

    var t = 0
    while (t < stride * nFrame) {
      val _t = t
      results(_t) = Engine.model.invoke(() => {
        val gradInputOffset = (_t / stride) * dim * stride + _t % stride
        val outputOffset = (_t / stride) * dim * stride + _t % stride
        val gradOutputOffset = (_t / stride) * dim * stride + _t % stride

        var sum = ev.fromType[Int](0)
        var d = 0
        while (d < dim) {
          sum = ev.plus(sum, ev.times(gradOutputArray(d * stride + gradOutputOffset),
            outputArray(d * stride + outputOffset)))
          d += 1
        }

        d = 0
        while (d < dim) {
          gradInputArray(d * stride + gradInputOffset) =
            ev.times(outputArray(d * stride + outputOffset),
              ev.minus(gradOutputArray(d * stride + gradOutputOffset), sum))
          d += 1
        }
      })

      t += 1
    }

    Engine.model.sync(results)

    gradInput
  }
}
