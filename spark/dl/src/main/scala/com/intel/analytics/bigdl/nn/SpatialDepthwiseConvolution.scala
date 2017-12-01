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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialDepthwiseConvolution[T: ClassTag](
  val nInputPlane: Int, val depthMultipler: Int,
  val kW: Int, val kH: Int,
  val sW: Int, val sH: Int,
  val padW: Int, val padH: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T]{

  require((padW >= 0 && padH >= 0) || (padW == -1 && padH == -1),
    s"Illegal padding configuration (padW: $padW, padH: $padH)")

  private val nOutputPlane = nInputPlane * depthMultipler

  private val weightShape = format match {
    case DataFormat.NCHW =>
      Array(nInputPlane, depthMultipler, kW, kH)
    case DataFormat.NHWC =>
      Array(kW, kH, nInputPlane, depthMultipler)
  }

  private val weight: Tensor[T] = Tensor[T](weightShape)
  private val gradWeight: Tensor[T] = Tensor[T](weightShape)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

  }
}

object SpatialDepthwiseConvolution {

}
