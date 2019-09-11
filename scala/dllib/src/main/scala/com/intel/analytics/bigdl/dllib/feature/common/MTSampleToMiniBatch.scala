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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef

import scala.reflect.ClassTag

/**
 * Convert a sequence of [[Sample]] to a sequence of [[MiniBatch]]
 * through function toMiniBatch using multi thread.
 */
class MTSampleToMiniBatch[A: ClassTag, T: ClassTag] (
          totalBatch: Int,
          transformer: Transformer[A, Sample[T]],
          miniBatch: Option[MiniBatch[T]] = None,
          featurePaddingParam: Option[PaddingParam[T]] = None,
          labelPaddingParam: Option[PaddingParam[T]] = None,
          partitionNum: Option[Int] = None)
      (implicit ev: TensorNumeric[T]) extends Transformer[A, MiniBatch[T]] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)
  var miniBatchBuffer = miniBatch.orNull
  private val batchSize = batchPerPartition
  private val sampleData = new Array[Sample[T]](batchSize)

  private val parallelism = EngineRef.getCoreNumber()

  private val transformers = (0 until parallelism).map(
    _ => transformer.cloneTransformer()
  ).toArray

  private val rawDataCache = new Array[Iterator[A]](batchSize)

  override def apply(prev: Iterator[A]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          // prefetch
          var count = 0
          while (count < batchSize && prev.hasNext) {
            val raw = prev.next()
            rawDataCache(count) = Iterator.single(raw)
            count += 1
          }

          // multi thread processing
          (0 until parallelism).toParArray.foreach{tid =>
            var j = tid
            while (j < count) {
              sampleData(j) = transformers(tid).apply(rawDataCache(j)).next()
              j += parallelism
            }
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

          if (count < batchSize) {
            miniBatchBuffer.set(sampleData.slice(0, count))
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

object MTSampleToMiniBatch {
  /**
   * Apply an MTSampleToMiniBatch transformer.
   *
   * @param batchSize           total batch size
   * @param transformer         transformer who rawData to Sample
   * @param featurePaddingParam feature padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @param labelPaddingParam   label padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @return
   */
  def apply[A: ClassTag, T: ClassTag](
         batchSize : Int,
         transformer: Transformer[A, Sample[T]],
         featurePaddingParam: Option[PaddingParam[T]] = None,
         labelPaddingParam: Option[PaddingParam[T]] = None,
         partitionNum: Option[Int] = None
         )(implicit ev: TensorNumeric[T]): MTSampleToMiniBatch[A, T] = {
    new MTSampleToMiniBatch[A, T](batchSize,
      transformer,
      None, featurePaddingParam, labelPaddingParam, partitionNum)
  }

  /**
   * Apply an MTSampleToMiniBatch transformer with UDF MiniBatch.
   *
   * @param batchSize total batch size
   * @param miniBatch An User-Defined MiniBatch to construct a mini batch.
   * @param transformer transformer who rawData to Sample
   * @return
   */
  def apply[A: ClassTag, T: ClassTag](
        miniBatch: MiniBatch[T],
        batchSize : Int,
        transformer: Transformer[A, Sample[T]],
        partitionNum: Option[Int])
        (implicit ev: TensorNumeric[T]): MTSampleToMiniBatch[A, T] = {
    new MTSampleToMiniBatch[A, T](batchSize,
      transformer,
      Some(miniBatch), partitionNum = partitionNum)
  }
}
