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
 * Cropping layer for 3D input (e.g. spatial or spatio-temporal).
 * It crops along spatial dimensions, i.e. depth,width and height.
 * Arguments:
 *  crop: could be int, Array[Int](3) or Array[Int](3,2)
 *    - If int: the same symmetric cropping
 *        is applied to dimension1, dimension2 and dimension3.
 *    - If Array of 3 ints:
 *       interpreted as three different
 *       symmetric cropping values for dim1, dim2 and dim3:
 *       (symmetric_dim1_crop, symmetric_dim2_crop,symmetric_dim3_crop ).
 *    - If Array of 3 Array of 2 ints:
 *       interpreted as
 *       ((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop),
  *       (left_dim3_crop, right_dim3_crop))
 *  dataFormat: A string,
 *           one of "channels_last" (default) or "channels_first".
 *           The ordering of the dimensions in the inputs.
 */
class Cropping3D[T: ClassTag]
( depthCrop: Array[Int],
  heightCrop: Array[Int],
  widthCrop: Array[Int],
  dataFormat: String)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(depthCrop(0) <= input.size(1)/2 &&
      heightCrop(0) <= input.size(2)/2 &&
      widthCrop(0) <= input.size(3)/2,
      "Cropping indices out of bounds.")
    require(depthCrop(1) <= input.size(1)/2 &&
      heightCrop(2) <= input.size(2)/2 &&
      widthCrop(3) <= input.size(3)/2,
      "Cropping indices out of bounds.")
    this.output = input
      .narrow(1, depthCrop(0), input.size(1) - depthCrop(0) - depthCrop(1))
      .narrow(2, heightCrop(0), input.size(1) - heightCrop(0) - heightCrop(1))
      .narrow(3, widthCrop(0), input.size(2) - widthCrop(0) - widthCrop(1))
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
    s"${getPrintName}(($depthCrop(0), $depthCrop(1))," +
      s"($heightCrop(0), $heightCrop(1))," +
      s"($widthCrop(0), $heightCrop(1))"
  }
}

object Cropping3D {
  def apply[T: ClassTag]( depthCrop: Array[Int] = Array[Int](0, 0),
                          heightCrop: Array[Int] = Array[Int](0, 0),
                          widthCrop: Array[Int] = Array[Int](0, 0),
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping3D[T] = {
    new Cropping3D[T](depthCrop, heightCrop, widthCrop, dataFormat)
  }

  def apply[T: ClassTag](
                          crop: Int = 0,
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping3D[T] = {
    new Cropping3D[T](
      Array[Int](crop, crop), Array[Int](crop, crop), Array[Int](crop, crop), dataFormat)
  }

  def apply[T: ClassTag]( symDepthCrop: Int,
                          symHeightCrop: Int,
                          symWidthCrop: Int,
                          dataFormat: String = "")
                        (implicit ev: TensorNumeric[T]): Cropping3D[T] = {
    new Cropping3D[T](Array[Int](symHeightCrop, symHeightCrop),
      Array[Int](symHeightCrop, symHeightCrop),
      Array[Int](symWidthCrop, symWidthCrop), dataFormat)
  }
}

