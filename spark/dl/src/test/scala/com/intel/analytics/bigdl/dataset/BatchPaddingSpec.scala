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

package com.intel.analytics.bigdl.dataset

import java.util

import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.Iterator
import scala.reflect.ClassTag
import scala.util.Random

class BatchPaddingSpec extends FlatSpec with Matchers with BeforeAndAfter {

  before {
    Engine.setNodeAndCore(1, 1)
  }

  "SampleToBatchPadding " should "be good when padding first dimension to same length" +
    "in one batch" in {
    val batchSize = 3
    val dictionaryLength = 5

    val input2 = Tensor[Float](3, dictionaryLength).apply1(e => Random.nextFloat())
    val target2 = Tensor(Storage(Array(2.0f)), 1, Array(1))
    val input1 = Tensor[Float](4, dictionaryLength).apply1(e => Random.nextFloat())
    val target1 = Tensor(Storage(Array(1.0f, 1.0f)), 1, Array(2))
    val input3 = Tensor[Float](2, dictionaryLength).apply1(e => Random.nextFloat())
    val target3 = Tensor(Storage(Array(3.0f, 1.0f, 2.0f)), 1, Array(3))

    val sample1 = new Sample[Float](input1, target1)
    val sample2 = new Sample[Float](input2, target2)
    val sample3 = new Sample[Float](input3, target3)

    val featurePadding = Tensor[Float](dictionaryLength).fill(100.0f)

    val trainData =
      Array[Sample[Float]](sample1, sample2, sample3, sample3, sample3, sample3)
    val trainSet = DataSet.array(trainData)
      .transform(SampleToBatch[Float](batchSize, Some(featurePadding), Some(10.0f)))

    val iter = trainSet.toLocal().data(train = false)

    val offset1 = new Array[Float](5)
    util.Arrays.fill(offset1, 0, 5, 100.0f)
    val offset2 = new Array[Float](10)
    util.Arrays.fill(offset2, 0, 10, 100.0f)

    val storage1 = input1.storage().array() ++ input2.storage().array() ++
      offset1 ++ input3.storage().array() ++ offset2
    val storage2 = input3.storage().array() ++ input3.storage().array() ++ input3.storage().array()

    val tensorInput1 = Tensor[Float](Storage(storage1), 1, Array(3, 4, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(1.0f, 1.0f, 10.0f, 2.0f, 10.0f, 10.0f, 3.0f, 1.0f, 2.0f)), 1, Array(3, 3))

    val tensorInput2 = Tensor[Float](Storage(storage2), 1, Array(3, 2, 5))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f)), 1, Array(3, 3))

    var batch = iter.next()
    var label = batch.labels
    var data = batch.data
    label should be (tensorTarget1)
    batch.data should be (tensorInput1)

    batch = iter.next()
    label = batch.labels
    data = batch.data
    label should be (tensorTarget2)
    batch.data should be (tensorInput2)
  }

  "SampleToBatchPadding " should "be good when padding to same length for all batch" in {
    val batchSize = 3
    val dictionaryLength = 5

    val input1 = Tensor[Float](3, dictionaryLength).apply1(e => Random.nextFloat())
    val target1 = Tensor(Storage(Array(2.0f)), 1, Array(1))
    val input2 = Tensor[Float](4, dictionaryLength).apply1(e => Random.nextFloat())
    val target2 = Tensor(Storage(Array(1.0f, 1.0f, 10.0f, 4.0f, 2.0f)), 1, Array(5))
    val input3 = Tensor[Float](2, dictionaryLength).apply1(e => Random.nextFloat())
    val target3 = Tensor(Storage(Array(3.0f, 1.0f, 4.0f)), 1, Array(3))

    val sample1 = new Sample[Float](input1, target1)
    val sample2 = new Sample[Float](input2, target2)
    val sample3 = new Sample[Float](input3, target3)

    val featurePadding = Tensor[Float](dictionaryLength).fill(100.0f)

    val trainData =
      Array[Sample[Float]](sample1, sample2, sample3, sample3, sample3, sample3)
    val trainSet = DataSet.array(trainData).transform(
      SampleToBatch[Float](batchSize, Some(featurePadding), Some(80.0f), Some(10)))

    val iter = trainSet.toLocal().data(train = false)

    val offset1 = new Array[Float](7 * dictionaryLength)
    util.Arrays.fill(offset1, 0, offset1.length, 100.0f)
    val offset2 = new Array[Float](6 * dictionaryLength)
    util.Arrays.fill(offset2, 0, offset2.length, 100.0f)
    val offset3 = new Array[Float](8 * dictionaryLength)
    util.Arrays.fill(offset3, 0, offset3.length, 100.0f)

    val storage1 = input1.storage().array() ++ offset1 ++ input2.storage().array() ++
      offset2 ++ input3.storage().array() ++ offset3
    val storage2 = input3.storage().array() ++ offset3 ++ input3.storage().array() ++
      offset3 ++ input3.storage().array() ++ offset3

    val tensorInput1 = Tensor[Float](Storage(storage1), 1, Array(3, 10, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(2.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f,
        1.0f, 1.0f, 10.0f, 4.0f, 2.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f,
        3.0f, 1.0f, 4.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f)), 1, Array(3, 10))

    val tensorInput2 = Tensor[Float](Storage(storage2), 1, Array(3, 10, 5))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(3.0f, 1.0f, 4.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f,
        3.0f, 1.0f, 4.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f,
        3.0f, 1.0f, 4.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f)), 1, Array(3, 10))

    var batch = iter.next()
    var label = batch.labels
    var data = batch.data
    label should be (tensorTarget1)
    batch.data should be (tensorInput1)

    batch = iter.next()
    label = batch.labels
    data = batch.data
    label should be (tensorTarget2)
    batch.data should be (tensorInput2)
  }

  "SampleToBatchPadding " should "be same to SampleToBatch when no padding" in {
    val batchSize = 3
    val totalCount = 100
    val trainData = new Array[Sample[Float]](totalCount)
    var i = 0
    while (i < totalCount) {
      val input = Tensor[Float](3, 224, 224).apply1(e => Random.nextFloat())
      val label = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
      trainData(i) = new Sample[Float](input, label)
      i += 1
    }
    val trainSet1 = DataSet.array(trainData)
      .transform(SampleToBatch[Float](batchSize))
    val trainSet2 = DataSet.array(trainData)
      .transform(SampleToBatchNoPadding(batchSize))

    val data1 = trainSet1.toLocal().data(train = false)
    val data2 = trainSet2.toLocal().data(train = false)

    while (data1.hasNext && data2.hasNext) {
      val batch1 = data1.next()
      val batch2 = data2.next()
      batch1.data should be (batch2.data)
      batch1.labels should be (batch2.labels)
    }
    data1.hasNext should be (false)
    data2.hasNext should be (false)
  }

  "SampleToBatchPadding " should "be same to LabeledSentenceToSample and SampleToBatch " +
    "when padding" in {
    val batchSize = 3
    val totalCount = 9
    val trainData = new Array[LabeledSentence[Float]](totalCount)
    var i = 0
    var base = Array[Float](1.0f)
    while (i < totalCount) {
      val input = Array[Float](3998) ++ base
      val label = base ++ Array[Float](3999)
      trainData(i) = new LabeledSentence[Float](input, label)
      base = base ++ Array[Float](i + 1)
      i += 1
    }

    val dictionaryLength = 4001
    val trainMaxLength = 10
    val featurePadding = Tensor[Float](dictionaryLength).fill(0.0f)
    featurePadding(4000) = 1
    val trainSet1 = DataSet.array(trainData)
      .transform(LabeledSentenceToSample(dictionaryLength))
      .transform(SampleToBatch[Float]
        (batchSize, Some(featurePadding), Some(3999), Some(trainMaxLength)))

    val trainSet2 = DataSet.array(trainData)
      .transform(LabeledSentenceToSample(dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))
      .transform(SampleToBatchNoPadding(batchSize))

    val data1 = trainSet1.toLocal().data(train = false)
    val data2 = trainSet2.toLocal().data(train = false)

    while (data1.hasNext && data2.hasNext) {
      val batch1 = data1.next()
      val batch2 = data2.next()
      batch1.data should be (batch2.data)
      batch1.labels should be (batch2.labels)
    }
    data1.hasNext should be (false)
    data2.hasNext should be (false)
  }
}

object SampleToBatchNoPadding {
  def apply[T: ClassTag]
  (batchSize: Int)
  (implicit ev: TensorNumeric[T]): SampleToBatchNoPadding[T]
  = new SampleToBatchNoPadding[T](batchSize)
}

class SampleToBatchNoPadding[T: ClassTag]
(totalBatch: Int)
(implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null
      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var oneFeatureLength: Int = 0
      private var oneLabelLength: Int = 0
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            if(!sample.feature().isContiguous() || !sample.label().isContiguous()) {
              throw new IllegalArgumentException(
                "Only support contiguous tensor, pls use tensor.contiguous() before batching")
            }
            if (featureData == null) {
              oneFeatureLength = sample.feature().nElement()
              oneLabelLength = sample.label().nElement()
              featureSize = Array(1) ++ sample.feature().size
              labelSize = Array(1) ++ sample.label().size
              featureData = new Array[T](batchSize * oneFeatureLength)
              labelData = new Array[T](batchSize * oneLabelLength)
            }
            sample.copyFromLabel(labelData, i*oneLabelLength, oneLabelLength)
            sample.copyFromFeature(featureData, i*oneFeatureLength, oneFeatureLength)
            i += 1
          }
          featureSize(0) = i
          labelSize(0) = i
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
