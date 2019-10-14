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
import com.intel.analytics.bigdl.utils.{Engine, Shape}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the
 * elements of the n-dimensional output Tensor lie in the range (0, 1) and sum to 1.
 * Softmax is defined as: f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
 * where shift = max_i(x_i).
 */
@SerialVersionUID(- 7842335603491194236L)
class SoftMax[T: ClassTag](var pos: Int = 1)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  @transient
  private var results: Array[Future[Unit]] = null

  private def getPositiveDimension(input: Tensor[T]): Int = {
    val inputDim = input.nDimension() // data batch dim
    pos = if (pos <= 0) {
      inputDim + pos
    }
    else pos
    require(1 <= pos && pos <= input.nDimension(),
      s"Invalid position: $pos ." + s"input dimension ${input.nDimension()}")
    pos
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(1 <= input.nDimension() && input.nDimension() <= 4,
      "1D, 2D, 3D or 4D tensor expected" +
        s"input dimension ${input.nDimension()}")
    pos = getPositiveDimension(input)

    val (nFrame, stride) = if (input.nDimension() - pos == 0) {
      (1, 1)
    } else if (input.nDimension() - pos == 1) {
      (input.size(pos), 1)
    } else if (input.nDimension() - pos == 2) {
      (1, input.size(pos + 1) * input.size(pos + 2))
    } else {
      (input.size(pos), input.size(pos + 2) * input.size(pos + 3))
    }
    if (results == null || results.length != nFrame * stride) {
      results = new Array[Future[Unit]](nFrame * stride)
    }
    output.resizeAs(input)
    SoftMax.updateOutput[T](input, output, results, pos)
    output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    pos = getPositiveDimension(input)
    gradInput.resizeAs(output)
    SoftMax.updateGradInput[T](input, gradOutput, gradInput, output, results, pos)
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}
object SoftMax{

  def apply[@specialized(Float, Double) T: ClassTag](pos: Int = 1)
      (implicit ev: TensorNumeric[T]) : SoftMax[T] = {
    new SoftMax[T](pos)
  }
  // Notice: SoftMin will call this function
  private[nn] def updateOutput[T: ClassTag](input: Tensor[T], output: Tensor[T],
    results: Array[Future[Unit]], pos: Int = 1) (implicit ev: TensorNumeric[T]): Tensor[T] = {

    val (nFrame, dim, stride) = if (input.nDimension() - pos == 0) {
      (1, input.size(pos), 1)
    } else if (input.nDimension() - pos == 1) {
      (input.size(pos), input.size(pos + 1), 1)
    } else if (input.nDimension() -pos == 2) {
      (1, input.size(pos), input.size(pos + 1) * input.size(pos + 2))
    } else {
      (input.size(pos), input.size(pos + 1), input.size(pos + 2) * input.size(pos + 3))
    }

    val outputArray = output.storage().array()
    val inputArray = if (input.isContiguous()) {
      input.storage().array()
    } else {
      input.contiguous().storage().array()
    }
    val storageOffset = input.storageOffset() - 1

    var t = 0
    while (t < stride * nFrame) {
      val _t = t
      results(_t) = Engine.model.invoke(() => {
        val inputOffset = (_t / stride) * dim * stride + _t % stride + storageOffset
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
    results: Array[Future[Unit]], pos: Int = 1
    )(implicit ev: TensorNumeric[T]): Tensor[T] = {

    require(input.size().deep == gradOutput.size().deep,
      "input should have the same size with gradOutput" +
        s"inputsize ${input.size().deep} gradOutput ${gradOutput.size().deep}")
    val (nFrame, dim, stride) = if (output.nDimension() - pos == 0) {
      (1, output.size(pos), 1)
    } else if (output.nDimension() - pos == 1) {
      (output.size(pos), output.size(pos + 1), 1)
    } else if (output.nDimension() - pos == 2) {
      (1, output.size(pos), output.size(pos + 1) * output.size(pos + 2))
    } else {
      (output.size(pos), output.size(pos + 1), output.size(pos + 2) * output.size(pos + 3))
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
