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

import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Upsampling layer for 2D inputs.
 * Repeats the heights and widths of the data by size(0) and size(1) respectively.
 *
 * If input's dataformat is NCHW, then the size of output is (N, C, H * size(0), W * size(1))
 *
 * @param size tuple of 2 integers. The upsampling factors for heights and widths.
 * @param format DataFormat, NCHW or NHWC
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class UpSampling2D[T: ClassTag] (val size: Array[Int], val format: DataFormat = DataFormat.NCHW)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  require(size.length == 2, s"UpSampling2D's size should be an array containing" +
    s" 2 elements, but got ${size.mkString("x")}")
  require(size(0) > 0 && size(1) > 0, "UpSampling2D's size should be bigger than 0," +
    s"but got ${size.mkString("x")}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"UpSampling2D requires 4D input, but got input dim ${input.length}")
    format match {
      case DataFormat.NCHW =>
        Shape(input(0), input(1), input(2)*size(0), input(3)*size(1))
      case DataFormat.NHWC =>
        Shape(input(0), input(1)*size(0), input(2)*size(1), input(3))
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 4, "UpSampling2D only supports 4D input")
    require(input.isContiguous(), "input should be contiguous")

    format match {
      case DataFormat.NCHW =>
        UpSampling2D.updateOutputNchw(input, output, size)
      case DataFormat.NHWC =>
        UpSampling2D.updateOutputNhwc(input, output, size)
      case _ =>
        throw new IllegalArgumentException("UpSampling2D: unsupported data format.")
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(gradOutput.dim() == 4, "UpSampling2D only supports 4D gradOutput")
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    gradInput.resizeAs(input).zero()

    format match {
      case DataFormat.NCHW =>
        UpSampling2D.updateGradInputNchw(gradInput, gradOutput, size)
      case DataFormat.NHWC =>
        UpSampling2D.updateGradInputNhwc(gradInput, gradOutput, size)
      case _ =>
        throw new IllegalArgumentException("UpSampling2D: unsupported data format.")
    }

    gradInput
  }
}

object UpSampling2D {
  def apply[T: ClassTag](size: Array[Int], format: DataFormat = DataFormat.NCHW)
           (implicit ev: TensorNumeric[T]): UpSampling2D[T] = {
    new UpSampling2D(size, format)
  }

  protected def updateOutputNchw[T: ClassTag](
        input: Tensor[T],
        output: Tensor[T],
        size: Array[Int])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    val inputHeight = input.size(3)
    val inputWeight = input.size(4)
    val outputHeight = inputHeight * size(0)
    val outputWeight = inputWeight * size(1)
    output.resize(input.size(1), input.size(2), outputHeight, outputWeight)


    val inputData = input.storage().array()
    var inputOffset = input.storageOffset() - 1

    val outputData = output.storage().array()
    var outputOffset = output.storageOffset() - 1

    var i = 0
    while (i < input.size(1) * input.size(2)) {
      var rowIndex = 0
      while (rowIndex < input.size(3)) {
        var columnIndex = 0
        // copy column
        while (columnIndex < input.size(4)) {
          var colReplicate = 0
          while (colReplicate < size(1)) {
            outputData(outputOffset) = inputData(inputOffset)
            outputOffset += 1
            colReplicate += 1
          }
          inputOffset += 1
          columnIndex += 1
        }

        // copy row
        var rowReplicate = 1
        while (rowReplicate < size(0)) {
          ev.arraycopy(outputData, outputOffset - outputWeight,
            outputData, outputOffset + (rowReplicate - 1) * outputWeight, outputWeight)
          rowReplicate += 1
        }
        outputOffset += outputWeight * (size(0) - 1)

        rowIndex += 1
      }

      i += 1
    }

    output

  }

  protected def updateOutputNhwc[T: ClassTag](
        input: Tensor[T],
        output: Tensor[T],
        size: Array[Int])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    val inputHeight = input.size(2)
    val inputWeight = input.size(3)
    val outputHeight = inputHeight * size(0)
    val outputWeight = inputWeight * size(1)
    output.resize(input.size(1), outputHeight, outputWeight, input.size(4))

    val channel = input.size(4)
    val owc = outputWeight * channel

    val inputData = input.storage().array()
    var inputOffset = input.storageOffset() - 1

    val outputData = output.storage().array()
    var outputOffset = output.storageOffset() - 1

    var i = 0
    while (i < input.size(1)) {
      var rowIndex = 0
      while (rowIndex < input.size(2)) {
        var columnIndex = 0
        // copy column
        while (columnIndex < input.size(3)) {
          var colReplicate = 0
          while (colReplicate < size(1)) {
            ev.arraycopy(inputData, inputOffset,
              outputData, outputOffset + colReplicate * channel, channel)
            colReplicate += 1
          }
          outputOffset += channel * size(1)
          inputOffset += channel
          columnIndex += 1
        }

        // copy row
        var rowReplicate = 1
        while (rowReplicate < size(0)) {
          ev.arraycopy(outputData, outputOffset - owc, outputData,
            outputOffset + (rowReplicate - 1) * owc, owc)
          rowReplicate += 1
        }
        outputOffset += owc * (size(0) - 1)

        rowIndex += 1
      }

      i += 1
    }
    output
  }

  protected def updateGradInputNhwc[T: ClassTag](
        gradInput: Tensor[T],
        gradOutput: Tensor[T],
        size: Array[Int])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    val gradInputData = gradInput.storage().array()
    var gradInputOffset = gradInput.storageOffset() - 1

    val gradOutputData = gradOutput.storage().array()
    var gradOutputOffset = gradOutput.storageOffset() - 1

    val gradInputWidth = gradInput.size(4)
    val gradOutputWidth = gradOutput.size(4)

    val channel = gradInput.size(4)
    val ocw = gradOutput.size(3) * gradOutput.size(4)

    var i = 0
    while (i < gradInput.size(1)) {
      var rowIndex = 0
      while (rowIndex < gradInput.size(2)) {
        var colIndex = 0
        while (colIndex < gradInput.size(3)) {
          var rowReplicate = 0
          while (rowReplicate < size(0)) {
            var colReplicate = 0
            while (colReplicate < size(1)) {
              ev.axpy(channel, ev.one, gradOutputData,
                gradOutputOffset + channel * colReplicate + rowReplicate * ocw, 1,
                gradInputData, gradInputOffset, 1)
              colReplicate += 1
            }
            rowReplicate += 1
          }
          gradInputOffset += channel
          gradOutputOffset += size(1) * channel
          colIndex += 1
        }
        gradOutputOffset += (size(0) - 1) * ocw
        rowIndex += 1
      }

      i += 1
    }

    gradInput
  }

  protected def updateGradInputNchw[T: ClassTag](
        gradInput: Tensor[T],
        gradOutput: Tensor[T],
        size: Array[Int])(implicit ev: TensorNumeric[T]) : Tensor[T] = {

    val gradInputData = gradInput.storage().array()
    var gradInputOffset = gradInput.storageOffset() - 1

    val gradOutputData = gradOutput.storage().array()
    var gradOutputOffset = gradOutput.storageOffset() - 1

    val gradInputWidth = gradInput.size(4)
    val gradOutputWidth = gradOutput.size(4)

    var i = 0
    while (i < gradInput.size(1) * gradInput.size(2) * gradInput.size(3)) {
      var row = 0
      while (row < size(0)) {
        var col = 0
        while (col < size(1)) {
          ev.axpy(gradInputWidth, ev.one, gradOutputData,
            gradOutputOffset + col, size(1),
            gradInputData, gradInputOffset, 1)
          col += 1
        }
        gradOutputOffset += gradOutputWidth
        row += 1
      }
      gradInputOffset += gradInputWidth
      i += 1
    }

    gradInput
  }
}
