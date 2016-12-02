/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialBatchNormalization[T: ClassTag](
  nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true)(
  implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum, affine) {
  override val nDim = 4

  override def toString(): String = {
    s"nn.SpatialBatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true)
                                                    (implicit ev: TensorNumeric[T]) : SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput, eps, momentum, affine)
  }
}
