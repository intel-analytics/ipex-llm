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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


abstract class KerasModule[A <: Activity: ClassTag, B <: Activity: ClassTag,
T: ClassTag](inputShape: Array[Int] = null)(implicit ev: TensorNumeric[T])
  extends LaborAdapter[A, B, T] {

  override def getBatchInputShape(): Activity = {
    if (inputShape != null) {
      val batchInputShape = Array(-1) ++ inputShape
      Tensor(data = batchInputShape, shape = Array(batchInputShape.length))
    } else {
      this.labor.getBatchInputShape()
    }
  }

  override def computeBatchOutputShape(inputShape: Activity): Activity = {
    this.labor.computeBatchOutputShape(inputShape)
  }
}
