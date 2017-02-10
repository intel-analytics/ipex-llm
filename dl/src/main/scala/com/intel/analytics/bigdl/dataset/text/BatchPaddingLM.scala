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

package com.intel.analytics.bigdl.dataset.text

import java.util

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}

import scala.collection.Iterator
import scala.reflect.ClassTag

object BatchPaddingLM {
  def apply[T: ClassTag]
  (batchSize: Int,
   vocabLength: Int,
   fixDataLength: Option[Int] = None,
   fixLabelLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T]): BatchPaddingLM[T]
  = new BatchPaddingLM[T](batchSize, vocabLength, fixDataLength, fixLabelLength)
}

class BatchPaddingLM[T: ClassTag]
(totalBatch: Int,
 vocabLength: Int,
 fixDataLength: Option[Int] = None,
 fixLabelLength: Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[Array[LabeledSentence[T]], MiniBatch[T]] {

  override def apply(prev: Iterator[Array[LabeledSentence[T]]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null

      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          val sentenceData = prev.next()
          val batchLength = sentenceData.length
          val maxLength = sentenceData(batchLength - 1).dataLength()
          val dataLength = fixDataLength.getOrElse(maxLength)
          val labelLength = fixLabelLength.getOrElse(maxLength)

          // init
          if (featureData == null || featureData.length < dataLength * vocabLength) {
            featureData = new Array[T](batchSize * dataLength * vocabLength)
          }
          if (labelData == null || labelData.length < labelLength) {
            labelData = new Array[T](batchSize * labelLength)
          }
          ev.getType() match {
            case DoubleType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Double]], 0, featureData.length, 0.0)
                util.Arrays.fill(labelData.asInstanceOf[Array[Double]], 0, labelData.length, 0.0)
            case FloatType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Float]], 0, featureData.length, 0.0f)
                util.Arrays.fill(labelData.asInstanceOf[Array[Float]], 0, labelData.length, 0.0f)
            case _ => throw new
                UnsupportedOperationException(s"BatchPaddingLM: Only Float/Double supported")
          }

          featureSize = Array(batchLength, maxLength, vocabLength)
          labelSize = Array(batchLength, maxLength)

          // padding
          i = 0
          while (i < batchLength) {
            val sentence = sentenceData(i)
            val startTokenIndex = sentence.getData(0)
            val endTokenIndex = if (labelLength == 1) 0
            else ev.toType[Int](sentence.getLabel(sentence.labelLength - 1))

            var j = 0
            while (j < sentence.dataLength) {
              featureData(i * maxLength * vocabLength  + j * vocabLength +
                ev.toType[Int](sentence.getData(j))) = ev.fromType[Float](1.0f)
              labelData(i * maxLength + j) = ev.plus(sentence.label()(j), ev.fromType[Float](1.0f))
              j += 1
            }
            while (j < maxLength) {
              featureData(i * maxLength * vocabLength + j * vocabLength + endTokenIndex) =
                ev.fromType[Float](1.0f)
              labelData(i * maxLength + j) = ev.plus(startTokenIndex, ev.fromType[Float](1.0f))
              j += 1
            }
            i += 1
          }

          featureTensor.set(Storage[T](featureData),
            storageOffset = 1, sizes = featureSize)
          labelTensor.set(Storage[T](labelData),
            storageOffset = 1, sizes = labelSize)
          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
