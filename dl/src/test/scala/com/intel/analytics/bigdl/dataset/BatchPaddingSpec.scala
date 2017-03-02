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

import com.intel.analytics.bigdl.dataset.text.{BatchPaddingLM, GroupSentence, LabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class BatchPaddingSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "BatchPadding " should "be good for language model when batchsize > 1" in {
    val input1 = Array(0.0f, 2.0f, 3.0f)
    val target1 = Array(2.0f, 3.0f, 4.0f)
    val input2 = Array(0.0f, 1.0f, 0.0f, 2.0f)
    val target2 = Array(1.0f, 0.0f, 2.0f, 4.0f)
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
    val trainSet = DataSet.array(GroupSentence(batchSize, trainData))
      .transform(BatchPaddingLM(batchSize = batchSize, dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))

    val iter = trainSet.toLocal().data(train = false)
    val tensorInput = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0,
            0, 0, 0, 1.0f, 0,
            0, 0, 0, 0, 1.0f,
            0, 0, 0, 0, 1.0f,
            1.0f, 0, 0, 0, 0,
            0, 0, 1.0f, 0, 0,
            0, 0, 0, 1.0f, 0,
            0, 0, 0, 0, 1.0f,
            1.0f, 0, 0, 0, 0,
            0, 1.0f, 0, 0, 0,
            1.0f, 0, 0, 0, 0,
            0, 0, 1.0f, 0, 0)), 1, Array(3, 4, 5))
    val tensorTarget = Tensor[Float](Storage(
      Array(4.0f, 5.0f, 1.0f, 1.0f,
            3.0f, 4.0f, 5.0f, 1.0f,
            2.0f, 1.0f, 3.0f, 5.0f)), 1, Array(3, 4))

    val batch = iter.next()
    batch.labels should be (tensorTarget)
    batch.data should be (tensorInput)
  }

  "BatchPadding " should "be good when batchsize = 1" in {
    val trainData = new Array[LabeledSentence[Float]](4)
    var i = 0
    var base = Array(1.0f)
    while (i < 4) {
      val input = Array(0.0f) ++ base ++ Array(100.0f)
      val label = Array(1.0f) ++ base ++ Array(100.0f)
      trainData(i) = new LabeledSentence[Float](input, label)
      i += 1
      base = base ++ Array(i.toFloat)
    }

    val trainMaxLength = 20
    val batchSize = 1
    val dictionaryLength = 4001
    val trainSet1 = DataSet.array(trainData.sortBy(_.dataLength()))
      .transform(LabeledSentenceToSample(dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))
      .transform(SampleToBatch(batchSize = batchSize))

    val trainSet2 = DataSet.array(GroupSentence(batchSize, trainData))
      .transform(BatchPaddingLM(batchSize = batchSize, dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))

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

      input1.length should be(input2.length)
      i = 0
      while (i < dictionaryLength * length) {
        input1(i) should be (input2(i))
        i += 1
      }
    }
    data1.hasNext should be (false)
    data2.hasNext should be (false)
  }
}
