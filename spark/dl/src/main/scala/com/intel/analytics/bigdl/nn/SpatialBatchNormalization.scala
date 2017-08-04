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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This file implements Batch Normalization as described in the paper:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe, Christian Szegedy
 * This implementation is useful for inputs coming from convolution layers.
 * For non-convolutional layers, see [[BatchNormalization]]
 * The operation implemented is:
 *
 *         ( x - mean(x) )
 * y = -------------------- * gamma + beta
 *      standard-deviation(x)
 *
 * where gamma and beta are learnable parameters.
 * The learning of gamma and beta is optional.
 */
@SerialVersionUID(- 9106336963903528047L)
class SpatialBatchNormalization[T: ClassTag](
  nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true,
  initWeight: Tensor[T] = null,
  initBias: Tensor[T] = null,
  initGradWeight: Tensor[T] = null,
  initGradBias: Tensor[T] = null)(
  implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum, affine,
    initWeight, initBias, initGradWeight, initGradBias) {
  override val nDim = 4

  override def toString(): String = {
    s"${getPrintName}[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
      nOutput: Int,
      eps: Double = 1e-5,
      momentum: Double = 0.1,
      affine: Boolean = true,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null)(implicit ev: TensorNumeric[T])
  : SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
      initWeight, initBias, initGradWeight, initGradBias)
  }
}
