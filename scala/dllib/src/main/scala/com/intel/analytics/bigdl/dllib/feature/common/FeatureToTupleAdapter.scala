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

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class FeatureToTupleAdapter[F, T](
    sampleTransformer: Transformer[(F, Any), Sample[T]]
  )(implicit ev: TensorNumeric[T]) extends Preprocessing[F, Sample[T]] {

  override def apply(prev: Iterator[F]): Iterator[Sample[T]] = {
    sampleTransformer.apply(prev.map(f => (f, None)))
  }
}

object FeatureToTupleAdapter {
  def apply[F, L, T: ClassTag](
      sampleTransformer: Transformer[(F, Any), Sample[T]]
    )(implicit ev: TensorNumeric[T]): FeatureToTupleAdapter[F, T] =
    new FeatureToTupleAdapter(sampleTransformer)
}
