/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.commons.lang3.SerializationUtils

import scala.collection.Iterator

/**
 * Transform data from type A to type B. It is usually used in data pre-process stage
 * @tparam A
 * @tparam B
 */
trait Transformer[A, B] extends Serializable {
  def apply(prev: Iterator[A]): Iterator[B]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: Transformer[B, C]): Transformer[A, C] = {
    new ChainedTransformer(this, other)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  def cloneTransformer(): Transformer[A, B] = {
    SerializationUtils.clone(this)
  }
}

class ChainedTransformer[A, B, C](first: Transformer[A, B], last: Transformer[B, C])
  extends Transformer[A, C] {
  override def apply(prev: Iterator[A]): Iterator[C] = {
    last(first(prev))
  }
}

object Identity {
  def apply[A](): Identity[A] = new Identity[A]()
}

class Identity[A] extends Transformer[A, A] {
  override def apply(prev: Iterator[A]): Iterator[A] = {
    prev
  }
}

object SampleToBatch {
  def apply[T](batchSize: Int): SampleToBatch[T]
  = new SampleToBatch[T](batchSize)
}

class SampleToBatch[T](totalBatch: Int)
  extends Transformer[Sample[T], MiniBatch[T]] {

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private val featureTensor: Tensor[T] = Tensor[T]()
      private val labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private val batchSize = Utils.getBatchSize(totalBatch)
      private var dimension: (Array[Int], Array[Int]) = null
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val smpl = prev.next()
            val featureLength = smpl.getFeature().size.reduceLeft(_*_)
            val labelLength = smpl.getLabel().size.reduceLeft(_*_)
            dimension = (smpl.getFeature().size, smpl.getLabel().size)
            if (featureData == null || featureData.length < batchSize * featureLength) {
              featureData = new Array[T](batchSize * featureLength)
            }
            if (labelData == null || labelData.length < batchSize * labelLength) {
              labelData = new Array[T](batchSize * labelLength)
            }
            smpl.copyToFeature(featureData, i*featureLength, featureLength)
            smpl.copyToLabel(labelData, i*labelLength, labelLength)
            i += 1
          }

          featureTensor.set(Storage[T](featureData),
            storageOffset = 1, sizes = Array(Array(i), dimension._1).flatten)
          labelTensor.set(Storage[T](labelData),
            storageOffset = 1, sizes = Array(Array(i), dimension._2).flatten)

          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
