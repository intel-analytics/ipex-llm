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

import java.util

import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class BatchPaddingSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "SampleToBatchPaddingLM " should "be good when batchsize = 1" in {
    val totalCount = 100
    val trainData = new Array[LabeledSentence[Float]](totalCount)
    var i = 0
    var base = Array(1.0f)
    while (i < totalCount) {
      val input = Array(0.0f) ++ base ++ Array(10.0f)
      val label = Array(1.0f) ++ base ++ Array(100.0f)
      trainData(i) = new LabeledSentence[Float](input, label)
      i += 1
      base = base ++ Array(i.toFloat)
    }
    val trainMaxLength = 20
    val batchSize = 1
    val dictionaryLength = 4001
    val trainDataReverse = trainData.reverse
    val trainSet1 = DataSet.array(trainDataReverse)
      .transform(LabeledSentenceToSample(dictionaryLength))
      .transform(SampleToBatch(batchSize = batchSize))

    val trainSet2 = DataSet.array(trainDataReverse)
      .transform(LabeledSentenceToSample(dictionaryLength))
      .transform(SampleToBatchPaddingLM[Float](batchSize, 0, 100, true))

    val data1 = trainSet1.toLocal().data(train = false)
    val data2 = trainSet2.toLocal().data(train = false)

    while (data1.hasNext && data2.hasNext) {
      val batch1 = data1.next()
      val input1 = batch1.data.storage().array()
      val label1 = batch1.labels.storage().array()

      val batch2 = data2.next()
      val input2 = batch2.data.storage().array()
      val label2 = batch2.labels.storage().array()
      val length = batch2.labels.size().product

      var i = 0
      while (i < length) {
        label1(i) should be (label2(i))
        i += 1
      }

      i = 0
      while (i < dictionaryLength * length) {
        input1(i) should be (input2(i))
        i += 1
      }
    }
    data1.hasNext should be (false)
    data2.hasNext should be (false)
  }

  "SampleToBatchPaddingLM " should "be good when padding to same length with batchsize > 1" in {
    val totalCount = 100
    val trainData = new Array[LabeledSentence[Float]](totalCount)
    var i = 0
    var base = Array(1.0f)
    while (i < totalCount) {
      val input = Array(0.0f) ++ base ++ Array(10.0f)
      val label = Array(1.0f) ++ base ++ Array(100.0f)
      trainData(i) = new LabeledSentence[Float](input, label)
      i += 1
      base = base ++ Array(i.toFloat)
    }

    val trainMaxLength = 110
    val batchSize = 2
    val dictionaryLength = 4001
    val trainSet1 = DataSet.array(trainData)
      .transform(LabeledSentenceToSample(dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))
      .transform(SampleToBatch(batchSize = batchSize))

    val trainSet2 = DataSet.array(trainData)
      .transform(LabeledSentenceToSample(dictionaryLength))
      .transform(SampleToBatchPaddingLM(batchSize, 0, 100, true, Some(0), Some(trainMaxLength)))

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

  "SampleToBatchPaddingLM " should "be good when batchsize > 1" in {
    val input2 = Array(0.0f, 2.0f, 3.0f)
    val target2 = Array(2.0f, 3.0f, 4.0f)
    val input1 = Array(0.0f, 1.0f, 0.0f, 2.0f)
    val target1 = Array(1.0f, 0.0f, 2.0f, 4.0f)
    val input3 = Array(0.0f, 3.0f)
    val target3 = Array(3.0f, 4.0f)
    val labeledSentence1 = new LabeledSentence[Float](input1, target1)
    val labeledSentence2 = new LabeledSentence[Float](input2, target2)
    val labeledSentence3 = new LabeledSentence[Float](input3, target3)

    val trainData =
      Array[LabeledSentence[Float]](labeledSentence1, labeledSentence2, labeledSentence3)
    val trainMaxLength = 5
    val batchSize = 5
    val dictionaryLength = 5
    val trainSet = DataSet.array(trainData)
      .transform(LabeledSentenceToSample(dictionaryLength))
      .transform(SampleToBatchPaddingLM[Float](batchSize, 0, 4, true))

    val iter = trainSet.toLocal().data(train = false)
    val tensorInput = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0,
        0, 1.0f, 0, 0, 0,
        1.0f, 0, 0, 0, 0,
        0, 0, 1.0f, 0, 0,
        1.0f, 0, 0, 0, 0,
        0, 0, 1.0f, 0, 0,
        0, 0, 0, 1.0f, 0,
        0, 0, 0, 0, 1.0f,
        1.0f, 0, 0, 0, 0,
        0, 0, 0, 1.0f, 0,
        0, 0, 0, 0, 1.0f,
        0, 0, 0, 0, 1.0f)), 1, Array(3, 4, 5))
    val tensorTarget = Tensor[Float](Storage(
      Array(2.0f, 1.0f, 3.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 1.0f,
        4.0f, 5.0f, 1.0f, 1.0f)), 1, Array(3, 4))

    val batch = iter.next()
    batch.labels should be (tensorTarget)
    batch.data should be (tensorInput)
  }

  "SampleToBatchPaddingLM " should "be same to SampleToBatch when no padding" in {
    val batchSize = 3
    val dictionaryLength = 5
    val totalCount = 100
    val trainData = new Array[Sample[Float]](totalCount)
    var i = 0
    var base = Array(1.0f)
    while (i < totalCount) {
      val input = Tensor[Float](10, dictionaryLength, 100).apply1(e => Random.nextFloat())
      val label = Tensor[Float](10, 2).apply1(e => Random.nextFloat())
      trainData(i) = new Sample[Float](input, label)
      i += 1
    }
    val trainSet1 = DataSet.array(trainData)
      .transform(SampleToBatchPaddingLM[Float](batchSize))
    val trainSet2 = DataSet.array(trainData)
      .transform(SampleToBatch(batchSize))

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

  "SampleToBatchPadding " should "be good when padding to same length in one batch" in {
    val batchSize = 3
    val dictionaryLength = 5

    val input2 = Tensor[Float](3, dictionaryLength).apply1(e => Random.nextFloat())
    val target2 = Tensor(Storage(Array(2.0f, 1.0f)), 1, Array(2))
    val input1 = Tensor[Float](4, dictionaryLength).apply1(e => Random.nextFloat())
    val target1 = Tensor(Storage(Array(1.0f, 1.0f)), 1, Array(2))
    val input3 = Tensor[Float](2, dictionaryLength).apply1(e => Random.nextFloat())
    val target3 = Tensor(Storage(Array(3.0f, 1.0f)), 1, Array(2))

    val sample1 = new Sample[Float](input1, target1)
    val sample2 = new Sample[Float](input2, target2)
    val sample3 = new Sample[Float](input3, target3)

    val trainData =
      Array[Sample[Float]](sample1, sample2, sample3, sample3, sample3, sample3)
    val trainSet = DataSet.array(trainData)
      .transform(SampleToBatchPadding[Float](batchSize, true, Some(100.0f)))

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
      Array(1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f)), 1, Array(3, 2))

    val tensorInput2 = Tensor[Float](Storage(storage2), 1, Array(3, 2, 5))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f)), 1, Array(3, 2))

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
    val target1 = Tensor(Storage(Array(2.0f, 1.0f)), 1, Array(2))
    val input2 = Tensor[Float](4, dictionaryLength).apply1(e => Random.nextFloat())
    val target2 = Tensor(Storage(Array(1.0f, 1.0f)), 1, Array(2))
    val input3 = Tensor[Float](2, dictionaryLength).apply1(e => Random.nextFloat())
    val target3 = Tensor(Storage(Array(3.0f, 1.0f)), 1, Array(2))

    val sample1 = new Sample[Float](input1, target1)
    val sample2 = new Sample[Float](input2, target2)
    val sample3 = new Sample[Float](input3, target3)

    val trainData =
      Array[Sample[Float]](sample1, sample2, sample3, sample3, sample3, sample3)
    val trainSet = DataSet.array(trainData)
      .transform(SampleToBatchPadding[Float](batchSize, true, Some(100.0f), Some(10)))

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
      Array(2.0f, 1.0f, 1.0f, 1.0f, 3.0f, 1.0f)), 1, Array(3, 2))

    val tensorInput2 = Tensor[Float](Storage(storage2), 1, Array(3, 10, 5))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f)), 1, Array(3, 2))

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
    val dictionaryLength = 5
    val totalCount = 100
    val trainData = new Array[Sample[Float]](totalCount)
    var i = 0
    var base = Array(1.0f)
    while (i < totalCount) {
      val input = Tensor[Float](3, 224, 224).apply1(e => Random.nextFloat())
      val label = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
      trainData(i) = new Sample[Float](input, label)
      i += 1
    }
    val trainSet1 = DataSet.array(trainData)
      .transform(SampleToBatchPadding[Float](batchSize, false))
    val trainSet2 = DataSet.array(trainData)
      .transform(SampleToBatch(batchSize))

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
