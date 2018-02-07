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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 *  Spatial 3D version of Dropout.
 *  This version performs the same function as Dropout, however it drops
 *  entire 3D feature maps instead of individual elements. If adjacent voxels
 *  within feature maps are strongly correlated (as is normally the case in
 *  early convolution layers) then regular dropout will not regularize the
 *  activations and will otherwise just result in an effective learning rate
 *  decrease. In this case, SpatialDropout3D will help promote independence
 *  between feature maps and should be used instead.
 *
 * @param p float between 0 and 1. Fraction of the input units to drop.
 * @param format  'NCHW' or 'NHWC'.
 *                 In 'NCHW' mode, the channels dimension (the depth)
 *                 is at index 1, in 'NHWC' mode is it at index 4.
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class SpatialDropout3D[T: ClassTag](
   val p: Double = 0.5,
   val format: String = "CHANNEL_FIRST",
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(format.toLowerCase() == "channel_first" || format.toLowerCase() == "channel_last",
    s"$format is not supported")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {

    val dimOrdering = if (format.toLowerCase() == "channel_first") DataFormat.NCHW else DataFormat.NHWC

    val layer = com.intel.analytics.bigdl.nn.SpatialDropout3D(
      initP = p,
      format = dimOrdering)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SpatialDropout3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double = 0.5,
    format: String = "CHANNEL_FIRST",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : SpatialDropout3D[T] = {
    new SpatialDropout3D[T](p, format, inputShape)
  }
}
