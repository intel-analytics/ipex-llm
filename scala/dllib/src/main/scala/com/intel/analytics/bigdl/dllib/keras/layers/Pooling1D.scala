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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import scala.reflect.ClassTag

/**
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'AveragePooling2D' and 'MaxPooling2D' instead.
 */
abstract class Pooling1D[T: ClassTag](
    val poolLength: Int = 2,
    val stride: Int = -1,
    val borderMode: String = "valid",
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends LayerWrapperByForward[T](KerasUtils.addBatch(inputShape)) {
  if (borderMode!=null) {
    require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
      s"Pooling1D: $borderMode")
  }
  val strideValue: Int = if (stride == -1) poolLength else stride
}
