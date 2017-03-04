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

object SampleToBatch {
  def apply[T: ClassTag]
  (totalBatch : Int,
   padFeatureValue : Option[Tensor[T]] = None,
   padLabelValue : Option[T] = None,
   fixLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T]): SampleToBatch[T]
  = new SampleToBatch[T](totalBatch, padFeatureValue, padLabelValue, fixLength)
}

/**
 * convert sample to MiniBatch, and you can choose padding feature or label
 * to same length in a batch
 * @param totalBatch batchsize
 * @param padFeatureValue  padding feature value if do padding
 * @param padLabelValue padding label value if do padding
 * @param fixLength padding fix length, default the max first dimension length of feature
 *                  or label in a batch
 */
class SampleToBatch[T: ClassTag]
(totalBatch : Int,
 padFeatureValue : Option[Tensor[T]] = None,
 padLabelValue : Option[T] = None,
 fixLength: Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  private def paddingTensor(data: Array[T], padValue: Tensor[T], start: Int, end: Int): Unit = {
    var offset = start
    val padArr = padValue.storage().array()
    while (offset < end) {
      val length = math.min(end - offset, padArr.length)
      System.arraycopy(padArr, 0, data, offset, length)
      offset += length
    }
  }

  private def paddingValue(data: Array[T], padValue: T, start: Int, end: Int): Unit = {
    ev.getType() match {
      case DoubleType =>
        util.Arrays.fill(data.asInstanceOf[Array[Double]], start, end, ev.toType[Double](padValue))
      case FloatType =>
        util.Arrays.fill(data.asInstanceOf[Array[Float]], start, end, ev.toType[Float](padValue))
      case _ => throw new UnsupportedOperationException(
        "SampleToBatchPadding: Only Float/Double supported")
    }
  }

  private def getSize(padLength: Int, batchLength: Int, tensorSize: Array[Int]): Int = {
    require(tensorSize.length >= 2)
    tensorSize(0) = 1
    tensorSize(1) = padLength
    val tensorLength = tensorSize.product
    tensorSize(0) = batchLength
    tensorLength
  }

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private val featureTensor: Tensor[T] = Tensor[T]()
      private val labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private var sampleData: Array[Sample[T]] = null

      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var oneFeatureLength: Int = 0
      private var oneLabelLength: Int = 0
      private var padFeatureLength: Int = 0
      private var padLabelLength: Int = 0
      private val padFeature: Boolean = if (padFeatureValue.isEmpty) false else true
      private val padLabel: Boolean = if (padLabelValue.isEmpty) false else true
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          var (maxfeatureElement, labelIndex) = (0, 0)
          var (maxlabelElement, featureIndex) = (0, 0)
          var padLength: Int = 0
          if (sampleData == null) sampleData = new Array[Sample[T]](batchSize)
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            require(sample.feature().isContiguous() && sample.label().isContiguous(),
              "SampleToBatchPadding: Only support contiguous tensor, pls use " +
                "tensor.contiguous() before batching")
            if (sampleData(i) == null) sampleData(i) = Sample()
            if ((maxfeatureElement != 0) && (!padFeature)) {
              require(sample.feature().nElement() == maxfeatureElement,
                "SampleToBatchPadding: all feature size should be same when not padding feature")
            }
            if ((maxlabelElement != 0) && (!padLabel)) {
              require(sample.label().nElement() == maxlabelElement,
                "SampleToBatchPadding: all label size should be same when not padding label")
            }
            sampleData(i).copy(sample)
            if (sample.feature().nElement() > maxfeatureElement) {
              maxfeatureElement = sample.feature().nElement()
              featureIndex = i
            }
            if (sample.label().nElement() > maxlabelElement) {
              maxlabelElement = sample.label().nElement()
              labelIndex = i
            }
            i += 1
          }
          val batchLength = i
          if (featureSize == null) {
            featureSize = Array(1) ++ sampleData(featureIndex).feature().size()
            labelSize = Array(1) ++ sampleData(labelIndex).label().size()
          }
          // init
          padFeatureLength = sampleData(featureIndex).feature().size(1)
          padLength = if (padFeature) fixLength.getOrElse(padFeatureLength) else padFeatureLength
          require(padLength >= padFeatureLength, "SampleToBatchPadding: fixLength " +
            "should not be less than first dimension of feature when padding")
          oneFeatureLength = getSize(padLength, batchLength, featureSize)

          padLabelLength = sampleData(labelIndex).label().size(1)
          padLength = if (padLabel) fixLength.getOrElse(padLabelLength) else padLabelLength
          require(padLength >= padLabelLength, "SampleToBatchPadding: fixLength " +
            "should not be less than first dimension of label when padding")
          oneLabelLength = getSize(padLength, batchLength, labelSize)

          if (featureData == null || featureData.length < batchSize * oneFeatureLength) {
            featureData = new Array[T](batchSize * oneFeatureLength)
          }
          if (labelData == null || labelData.length < batchSize * oneLabelLength) {
            labelData = new Array[T](batchSize * oneLabelLength)
          }

          // padding
          if (padFeature) {
            require(padFeatureValue.get.isContiguous(), "SampleToBatch: padFeatureValue " +
              "should be contiguous")
            require((padFeatureValue.get.dim() + 1) == sampleData(featureIndex).feature().dim(),
              "SampleToBatch: padFeatureValue dim should be Feature dim - 1")
            paddingTensor(featureData, padFeatureValue.get, 0, featureData.length)
          }
          if (padLabel) {
            paddingValue(labelData, padLabelValue.get, 0, labelData.length)
          }

          i = 0
          while (i < batchLength) {
            val sample = sampleData(i)
            sample.copyFromFeature(featureData, i * oneFeatureLength, sample.feature().nElement())
            sample.copyFromLabel(labelData, i * oneLabelLength, sample.label().nElement())
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
