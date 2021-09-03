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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Abstract class for different global pooling 2D layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'GlobalAveragePooling2D' and 'GlobalMaxPooling2D' instead.
 */
abstract class GlobalPooling2D[T: ClassTag](
   val dimOrdering: DataFormat = DataFormat.NCHW,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"GlobalPooling2D requires 4D input, but got input dim ${input.length}")
    dimOrdering match {
      case DataFormat.NCHW => Shape(input(0), input(1))
      case DataFormat.NHWC => Shape(input(0), input(3))
    }
  }

}
