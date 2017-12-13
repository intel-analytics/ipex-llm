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

import scala.reflect.ClassTag

/**
 * Cropping layer for 2D input (e.g. picture).
 * It crops along spatial dimensions, i.e. width and height.
 * # Input shape
 *     4D tensor with shape:
 *      `(batchSize, channels, first_axis_to_crop, second_axis_to_crop)`
 * # Output shape
 *      4D tensor with shape:
 *      `(batchSize, channels, first_cropped_axis, second_cropped_axis)`
 *
 * @param heightCrop Array of length 2. How many units should be trimmed off at the beginning
 *                   and end of the height dimension.
 * @param widthCrop Array of length 2. How many units should be trimmed off at the beginning
 *                  and end of the width dimension
 * @param dataFormat: DataFormat.NCHW or DataFormat.NHWC
 */
@SerialVersionUID(3462228835945094156L)
class Cropping2D[T: ClassTag](
    val heightCrop: Array[Int],
    val widthCrop: Array[Int],
    val dataFormat: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 4, "input dimensions should be 4." +
      " (batchSize, channels, first_axis_to_crop, second_axis_to_crop)")

    val (hdim, wdim, hStart, lenHCropped, wStart, lenWCropped) = calculateStartAndLength(input)

    require(lenHCropped > 0, s"heightCrop: ${heightCrop.mkString(", ")} is too large. Height" +
      s" dimension length: ${input.size(hdim)}")
    require(lenWCropped > 0, s"widthCrop: ${widthCrop.mkString(", ")} is too large. Width" +
      s" dimension length: ${input.size(wdim)}")

    this.output = input
      .narrow(hdim, hStart, lenHCropped)
      .narrow(wdim, wStart, lenWCropped)
      .contiguous()
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val (hdim, wdim, hStart, lenHCropped, wStart, lenWCropped) = calculateStartAndLength(input)
    gradInput.resizeAs(input).zero()
      .narrow(hdim, hStart, lenHCropped)
      .narrow(wdim, wStart, lenWCropped)
      .copy(gradOutput)
  }

  /**
   * Calculate the start position and length after cropping
   */
  private def calculateStartAndLength(input: Tensor[T]): (Int, Int, Int, Int, Int, Int) = {
    val (hdim, wdim) = dataFormat match {
      case DataFormat.NCHW => (3, 4)
      case DataFormat.NHWC => (2, 3)
      case _ => throw new IllegalArgumentException(s"$dataFormat is not a supported format")
    }

    val hStart = heightCrop(0) + 1
    val lenHCropped = input.size(hdim) - heightCrop(0) - heightCrop(1)
    val wStart = widthCrop(0) + 1
    val lenWCropped = input.size(wdim) - widthCrop(0) - widthCrop(1)
    (hdim, wdim, hStart, lenHCropped, wStart, lenWCropped)
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"$getPrintName(heightCrop: ${heightCrop.mkString(", ")};" +
      s" widthCrop: ${widthCrop.mkString(", ")}.)"
  }
}

object Cropping2D {
  def apply[T: ClassTag](
      heightCrop: Array[Int],
      widthCrop: Array[Int],
      format: DataFormat = DataFormat.NCHW) (implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    new Cropping2D[T](heightCrop, widthCrop, format)
  }
}
