/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.vfl.utils

import com.intel.analytics.bigdl.dllib.feature._
import com.intel.analytics.bigdl.dllib.feature.dataset.{MiniBatch, PaddingParam, Sample, SparseMiniBatch, TensorSample, Transformer}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.collection.Iterator
import scala.reflect.ClassTag

/**
 * Convert a sequence of [[Sample]] to a sequence of [[MiniBatch]] through function toMiniBatch.
 */
class SampleToMiniBatch[T: ClassTag] private[bigdl](
                                                   batchSize: Int,
                                                   miniBatch: Option[MiniBatch[T]] = None,
                                                   featurePaddingParam: Option[PaddingParam[T]] = None,
                                                   labelPaddingParam: Option[PaddingParam[T]] = None,
                                                   partitionNum: Option[Int] = None)
                                                 (implicit ev: TensorNumeric[T]) extends Transformer[Sample[T], MiniBatch[T]] {

  var miniBatchBuffer = miniBatch.orNull
  protected val sampleData = new Array[Sample[T]](batchSize)

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            sampleData(i) = sample
            i += 1
          }
          if (null == miniBatchBuffer) {
            val firstSample = sampleData(0)
            miniBatchBuffer = if (firstSample.isInstanceOf[TensorSample[T]]) {
              SparseMiniBatch(firstSample.numFeature(), firstSample.numLabel())
            } else {
              MiniBatch(firstSample.numFeature(), firstSample.numLabel(),
                featurePaddingParam, labelPaddingParam)
            }
          }

          if (i < batchSize) {
            miniBatchBuffer.set(sampleData.slice(0, i))
          } else {
            miniBatchBuffer.set(sampleData)
          }
        } else {
          null
        }
      }
    }
  }
}

object SampleToMiniBatch {
  /**
   * Apply an SampleToMiniBatch transformer.
   *
   * @param batchSize           total batch size
   * @param featurePaddingParam feature padding strategy, see
   *                            [[com.intel.analytics.bigdl.dllib.feature.dataset.PaddingParam]] for details.
   * @param labelPaddingParam   label padding strategy, see
   *                            [[com.intel.analytics.bigdl.dllib.feature.dataset.PaddingParam]] for details.
   * @return
   */
  def apply[T: ClassTag](
                          batchSize : Int,
                          featurePaddingParam: Option[PaddingParam[T]] = None,
                          labelPaddingParam: Option[PaddingParam[T]] = None,
                          partitionNum: Option[Int] = None
                        )(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, None, featurePaddingParam, labelPaddingParam, partitionNum)
  }

  /**
   * Apply an SampleToMiniBatch transformer with UDF MiniBatch.
   *
   * @param batchSize total batch size
   * @param miniBatch An User-Defined MiniBatch to construct a mini batch.
   * @return
   */
  def apply[T: ClassTag](
                          miniBatch: MiniBatch[T],
                          batchSize : Int,
                          partitionNum: Option[Int])(implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, Some(miniBatch), partitionNum = partitionNum)
  }
}
