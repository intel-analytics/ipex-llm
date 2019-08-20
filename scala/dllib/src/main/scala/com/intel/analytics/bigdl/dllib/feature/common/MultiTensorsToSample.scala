/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * a Preprocessing that converts multiple tensors to Sample.
 */
class MultiTensorsToSample[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Preprocessing[Array[Tensor[T]], Sample[T]] {

  override def apply(prev: Iterator[Array[Tensor[T]]]): Iterator[Sample[T]] = {
    prev.map(Sample(_))
  }
}

object MultiTensorsToSample {
  def apply[F, T: ClassTag]()(implicit ev: TensorNumeric[T]): MultiTensorsToSample[T] =
    new MultiTensorsToSample()
}
