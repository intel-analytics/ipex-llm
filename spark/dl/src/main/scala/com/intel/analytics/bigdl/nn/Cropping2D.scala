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
import scala.reflect.ClassTag

/**
 * Cropping layer for 2D input (e.g. picture).
 * It crops along spatial dimensions, i.e. width and height.
 * Arguments:
 *  crop: could be int, Array[Int](2) or Array[Int](2,2)
 *    - If int: the same symmetric cropping
 *               is applied to width and height.
 *    - If Array of 2 ints:
 *               interpreted as two different
 *               symmetric cropping values for height and width:
 *               (symmetric_height_crop, symmetric_width_crop).
 *    - If Array of 2 Array of 2 ints:
 *               interpreted as
 *               ((top_crop, bottom_crop), (left_crop, right_crop))
 *  dataFormat: A string,
 *           one of "channels_last" (default) or "channels_first".
 *           The ordering of the dimensions in the inputs.
 */
class Cropping2D[T: ClassTag]
(heightCrop: Array[Int],
 widthCrop: Array[Int],
 dataFormat: String)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val patchSize = Array[Int](heightCrop(1) - heightCrop(0) + 1, widthCrop(1) - widthCrop(0) + 1)
    require(heightCrop(0) <= input.size(1)/2 && widthCrop(0) <= input.size(2)/2,
      "Cropping indices out of bounds.")
    require(heightCrop(1) <= input.size(1)/2
      && widthCrop(1) <= input.size(2)/2, "Cropping indices out of bounds.")
    this.output = input.narrow(1, heightCrop(0), input.size(1) - heightCrop(0) - heightCrop(1))
      .narrow(2, widthCrop(0), input.size(2) - widthCrop(0) - widthCrop(1))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"${getPrintName}($heightCrop(0), $heightCrop(1), $widthCrop(0), $heightCrop(1))"
  }
}

object Cropping2D {
  def apply[T: ClassTag]( heightCrop: Array[Int] = Array[Int](0, 0),
                          widthCrop: Array[Int] = Array[Int](0, 0),
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    new Cropping2D[T](heightCrop, widthCrop, dataFormat)
  }

  def apply[T: ClassTag](
                          crop: Int = 0,
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    new Cropping2D[T](Array[Int](crop, crop), Array[Int](crop, crop), dataFormat)
  }

  def apply[T: ClassTag](
                          symHeightCrop: Int,
                          symWidthCrop: Int,
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    new Cropping2D[T](Array[Int](symHeightCrop, symHeightCrop),
      Array[Int](symWidthCrop, symWidthCrop), dataFormat)
  }
}
