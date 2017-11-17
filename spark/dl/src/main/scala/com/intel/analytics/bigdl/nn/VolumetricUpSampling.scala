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
 * Upsampling layer for 3D inputs.
 * Repeats the 1st, 2nd and 3rd dimensions
 * of the data by size[0], size[1] and size[2] respectively.
 * The input data is assumed to be of the form `minibatch x channels x depth x height x width`.
 * The input parameter scale_factor specifies the amount of upsampling
 * For nearest neighbour, output size will be:
 * odepth  = depth*scale_factor
 * owidth  = width*scale_factor
 * oheight  = height*scale_factor
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class VolumetricUpSampling[T: ClassTag](scaleFactor: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private val inputSize = new Array[Int](5)
  private val outputSize = new Array[Int](5)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 5, "only supports 5d tensors")
    val xdim = 5
    val ydim = 4
    val zdim = 3
    (0 until 5).foreach(i => {
      inputSize(i) = input.size(i + 1)
      outputSize(i) = input.size(i + 1)
    })
    outputSize(2) = outputSize(2) * scaleFactor
    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput
  }
}
