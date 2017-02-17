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

object SampleToBatchPaddingLM {
  def apply[T: ClassTag]
  (totalBatch : Int,
   startToken: Int = -1,
   endToken: Int = -1,
   padding : Boolean = false,
   paddingValue : Option[T] = None,
   paddingLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T]): SampleToBatchPaddingLM[T]
  = new SampleToBatchPaddingLM[T](
    totalBatch, startToken, endToken, padding, paddingValue, paddingLength)
}

/**
 * convert sample to MiniBatch, and you can choose to do padding or not.
 * Padding both feature and label to the same first dimension length.
 * @param totalBatch batchsize
 * @param startToken startToken for language model
 * @param endToken endToken for language model
 * @param padding whether padding to the same length in a batch
 * @param paddingValue padding value, default 0
 * @param paddingLength padding length, default the max first dimension length of feature in a batch
 */
class SampleToBatchPaddingLM[T: ClassTag]
(totalBatch : Int,
 startToken: Int = -1,
 endToken: Int = -1,
 padding : Boolean,
 paddingValue : Option[T] = None,
 paddingLength: Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private var sampleData: Array[Sample[T]] = null

      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var oneFeatureLength: Int = 0
      private var oneLabelLength: Int = 0
      private var firstDimLength: Int = 0
      private var vocabLength: Int = 0
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          featureSize = null
          if (sampleData == null) sampleData = new Array[Sample[T]](batchSize)
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            require(sample.feature().isContiguous() && sample.label().isContiguous(),
              "SampleToBatchPaddingLM: Only support contiguous tensor, pls use " +
                "tensor.contiguous() before batching")
            require(sample.feature().size(1) == sample.label().size(1),
              "SampleToBatchPaddingLM: first dimension for feature and label should " +
                "be same in language model")
            if (sampleData(i) == null) sampleData(i) = Sample()
            if (featureSize != null && !padding) {
              require(util.Arrays.equals(sample.feature().size(), featureSize),
                "SampleToBatchPaddingLM: all feature size should be same when no padding")
              require(util.Arrays.equals(sample.label().size(), labelSize),
                "SampleToBatchPaddingLM: all label size should be same when no padding")
            }
            sampleData(i).copy(sample)
            if ((featureSize == null) || (sample.feature().nElement() > featureSize.product)) {
              firstDimLength = sample.feature().size(1)
              vocabLength = sample.feature().size(2)
              featureSize = sample.feature().size()
              labelSize = sample.label().size()
            }
            i += 1
          }
          val batchLength = i

          // init
          val padLength = paddingLength.getOrElse(firstDimLength)
          require(padLength >= firstDimLength, "SampleToBatchPaddingLM: paddingLength should not " +
            "be less than first dimension of feature")
          oneFeatureLength = padLength * featureSize.product / firstDimLength
          oneLabelLength = padLength * labelSize.product / firstDimLength
          // batch size
          featureSize = Array(batchLength, padLength) ++ featureSize.slice(1, featureSize.length)
          labelSize = Array(batchLength, padLength) ++ labelSize.slice(1, labelSize.length)
          if (featureData == null || featureData.length < batchSize * oneFeatureLength) {
            featureData = new Array[T](batchSize * oneFeatureLength)
          }
          if (labelData == null || labelData.length < batchSize * oneLabelLength) {
            labelData = new Array[T](batchSize * oneLabelLength)
          }

          // padding
          if (padding) {
            val padValue = paddingValue.getOrElse(ev.zero)
            ev.getType() match {
              case DoubleType =>
                util.Arrays.fill(featureData.asInstanceOf[Array[Double]], 0, featureData.length,
                  ev.toType[Double](padValue))
                util.Arrays.fill(labelData.asInstanceOf[Array[Double]], 0, labelData.length,
                  ev.toType[Double](padValue))
              case FloatType =>
                util.Arrays.fill(featureData.asInstanceOf[Array[Float]], 0, featureData.length,
                  ev.toType[Float](padValue))
                util.Arrays.fill(labelData.asInstanceOf[Array[Float]], 0, labelData.length,
                  ev.toType[Float](padValue))
              case _ => throw new UnsupportedOperationException(
                "SampleToBatchPaddingLM: Only Float/Double supported")
            }
          }

          i = 0
          while (i < batchLength) {
            val sample = sampleData(i)
            sample.copyFromLabel(labelData, i * oneLabelLength, sample.label().nElement())
            sample.copyFromFeature(featureData, i * oneFeatureLength, sample.feature().nElement())

            // add start and end token
            var j = sample.label().size(1)
            while ((j < padLength) && (startToken >= 0) && (endToken >= 0)) {
              featureData(i * oneFeatureLength + j * vocabLength + endToken) = ev.one
              labelData(i * oneLabelLength + j) = ev.plus(ev.fromType(startToken), ev.one)
              j += 1
            }
            i += 1
          }
          featureTensor.set(Storage[T](featureData), storageOffset = 1, sizes = featureSize)
          labelTensor.set(Storage[T](labelData), storageOffset = 1, sizes = labelSize)
          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}