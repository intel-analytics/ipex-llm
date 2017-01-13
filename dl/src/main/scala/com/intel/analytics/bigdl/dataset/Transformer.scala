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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.commons.lang3.SerializationUtils

import scala.collection.Iterator
import scala.reflect.ClassTag

/**
 * Transform a data stream of type A to type B. It is usually used in data pre-process stage.
 * Different transformers can compose a pipeline. For example, if there're transformer1 from A to
 * B, transformer2 from B to C, and transformer3 from C to D, you can compose them into a bigger
 * transformer from A to D by transformer1 -> transformer2 -> transformer 3.
 *
 * The purpose of transformer is for code reuse. Many deep learning share many common data
 * pre-process steps. User needn't write them every time, but can reuse others work.
 *
 * Transformer can be used with RDD(rdd.mapPartition), iterator and DataSet.
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
  def apply[T: ClassTag]
  (batchSize: Int,
   tableOfBatch: Boolean = false)
  (implicit ev: TensorNumeric[T]): SampleToBatch[T]
  = new SampleToBatch[T](batchSize, tableOfBatch)
}

class SampleToBatch[T: ClassTag]
  (totalBatch: Int, tableOfBatch: Boolean = false)
  (implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private val featureTensor: Tensor[T] = Tensor[T]()
      private val labelTensor: Tensor[T] = Tensor[T]()
      private val featureTable: Table = T()
      private val labelTable: Table = T()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var _featureLength: Int = 0
      private var _labelLength: Int = 0
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            val featureLength = sample.feature().nElement()
            val labelLength = sample.label().nElement()
            if (featureSize == null || labelSize == null
              || _featureLength != featureLength
              || _labelLength != labelLength) {
              featureSize = Array(1) ++ sample.feature().size
              labelSize = Array(1) ++ sample.label().size
              _featureLength = featureLength
              _labelLength = labelLength
            }
            if (featureData == null || featureData.length < batchSize * featureLength) {
              featureData = new Array[T](batchSize * featureLength)
            }
            if (labelData == null || labelData.length < batchSize * labelLength) {
              labelData = new Array[T](batchSize * labelLength)
            }
            sample.copyToLabel(labelData, i*labelLength, labelLength)
            sample.copyToFeature(featureData, i*featureLength, featureLength)

            featureTable(i + 1) = Tensor()
              .set(Storage[T](featureData), storageOffset = i*featureLength + 1,
                sizes = sample.feature().size)
            labelTable(i + 1) = Tensor()
              .set(Storage[T](labelData), storageOffset = i*labelLength + 1,
                sizes = sample.label().size)
            i += 1
          }

          var j = i + 1
          while (j <= featureTable.length) {
            featureTable.remove(j)
            j += 1
          }

          j = i + 1
          while (j <= labelTable.length) {
            labelTable.remove(j)
            j += 1
          }

          featureSize(0) = i
          labelSize(0) = i
          featureTensor.set(Storage[T](featureData),
            storageOffset = 1, sizes = featureSize)
          labelTensor.set(Storage[T](labelData),
            storageOffset = 1, sizes = labelSize)
          tableOfBatch match {
            case true => MiniBatch(featureTable, labelTable)
            case false => MiniBatch(featureTensor, labelTensor)
          }
        } else {
          null
        }
      }
    }
  }
}
