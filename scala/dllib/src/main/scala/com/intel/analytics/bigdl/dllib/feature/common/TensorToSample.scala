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
package com.intel.analytics.bigdl.dllib.feature.common

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * a Preprocessing that converts Tensor to Sample.
 */
class TensorToSample[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Preprocessing[Tensor[T], Sample[T]] {

  override def apply(prev: Iterator[Tensor[T]]): Iterator[Sample[T]] = {
    prev.map(Sample(_))
  }
}

object TensorToSample {
  def apply[F, T: ClassTag]()(implicit ev: TensorNumeric[T]): TensorToSample[T] =
    new TensorToSample()
}

