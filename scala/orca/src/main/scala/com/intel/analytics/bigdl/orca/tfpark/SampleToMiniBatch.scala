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

package com.intel.analytics.bigdl.orca.tfpark

import com.intel.analytics.bigdl.dllib.feature.dataset._
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.collection.Iterator
import scala.reflect.ClassTag

class SampleToMiniBatch[T: ClassTag](
        totalBatch: Int,
        miniBatch: Option[MiniBatch[T]] = None,
        featurePaddingParam: Option[PaddingParam[T]] = None,
        labelPaddingParam: Option[PaddingParam[T]] = None,
        partitionNum: Option[Int] = None, dropRemainder: Boolean = false)
        (implicit ev: TensorNumeric[T]) extends Transformer[Sample[T], MiniBatch[T]] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)
  var miniBatchBuffer = miniBatch.orNull
  private val batchSize = batchPerPartition
  private val sampleData = new Array[Sample[T]](batchSize)

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {

      var batchValid = false

      override def hasNext: Boolean = {
        // data already fetched  but not consumed
        if (batchValid) return true

        successOnFetch()
      }

      // fetch the current batch, set it to miniBatchBuffer,
      // return success or not
      def successOnFetch(): Boolean = {
        if (!prev.hasNext) {
          return false
        }

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
          // do not have enough data to form a batch
          if (dropRemainder) {
            return false
          } else {
            miniBatchBuffer.set(sampleData.slice(0, i))
          }
        } else {
          miniBatchBuffer.set(sampleData)
        }
        batchValid = true
        true
      }

      override def next(): MiniBatch[T] = {
        if (hasNext) {
          // data consumed, set valid to false
          batchValid = false
          miniBatchBuffer
        } else {
          throw new NoSuchElementException("Next on a empty iterator")
        }
      }
    }
  }
}
