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
 * construct a Preprocessing that convert (Feature, Label) tuple to a Sample.
 * The returned Preprocessing is robust for the case label = None, in which the
 * Sample is derived from Feature only.
 * @param featureStep Preprocessing for feature, transform F to Tensor[T]
 * @param labelStep Preprocessing for label, transform L to Tensor[T]
 * @tparam F data type from feature column, E.g. Array[_] or Vector
 * @tparam L data type from label column, E.g. Float, Double, Array[_] or Vector
 */
class FeatureLabelPreprocessing[F, X, L, T: ClassTag] private[zoo] (
    featureStep: Preprocessing[F, X],
    labelStep: Preprocessing[L, Tensor[T]]
  )(implicit ev: TensorNumeric[T]) extends Preprocessing[(F, Option[L]), Sample[T]] {

  override def apply(prev: Iterator[(F, Option[L])]): Iterator[Sample[T]] = {
    prev.map { case (feature, label ) =>
      val featureTensors = featureStep(Iterator(feature)).next()
      featureTensors match {
        case ft: Tensor[T] =>
          val ft = featureTensors.asInstanceOf[Tensor[T]]
          label match {
            case Some(l) =>
              val labelTensor = labelStep(Iterator(l)).next()
              Sample[T](ft, labelTensor)
            case None =>
              Sample[T](ft)
          }
        case fat: Array[Tensor[T]] =>
          label match {
            case Some(l) =>
              val labelTensor = labelStep(Iterator(l)).next()
              Sample[T](fat, labelTensor)
            case None =>
              Sample[T](fat)
          }
        case _ =>
          throw new UnsupportedOperationException(
            s"FeatureLabelPreprocessing expects table or tensor, but got $featureTensors")
      }
    }
  }
}

object FeatureLabelPreprocessing {
  def apply[F, X, L, T: ClassTag](
      featureStep: Preprocessing[F, X],
      labelStep: Preprocessing[L, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): FeatureLabelPreprocessing[F, X, L, T] =
    new FeatureLabelPreprocessing(featureStep, labelStep)
}
