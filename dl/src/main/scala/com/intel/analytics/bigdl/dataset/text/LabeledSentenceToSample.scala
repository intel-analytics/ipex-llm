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

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}

import scala.collection.Iterator
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import java.util

object LabeledSentenceToSample {
  def apply(vocabLength: Int,
            fixDataLength: Option[Int] = None,
            fixLabelLength: Option[Int] = None)
  : LabeledSentenceToSample =
    new LabeledSentenceToSample(
      vocabLength,
      fixDataLength,
      fixLabelLength)
}

class LabeledSentenceToSample(vocabLength: Int,
                              fixDataLength: Option[Int],
                              fixLabelLength: Option[Int])
  extends Transformer[LabeledSentence[Float], Sample[Float]] {
  private val buffer = new Sample[Float]()
  private var featureBuffer: Array[Float] = null
  private var labelBuffer: Array[Float] = null

  override def apply(prev: Iterator[LabeledSentence[Float]]): Iterator[Sample[Float]] = {
    prev.map(sentence => {

      val dataLength = fixDataLength.getOrElse(sentence.dataLength())
      val labelLength = fixLabelLength.getOrElse(sentence.labelLength())

      if (featureBuffer == null || featureBuffer.length < dataLength * vocabLength) {
        featureBuffer = new Array[Float](dataLength * vocabLength)
      }
      if (labelBuffer == null || labelBuffer.length < labelLength) {
        labelBuffer = new Array[Float](labelLength)
      }

      // initialize featureBuffer to 0.0

      util.Arrays.fill(featureBuffer, 0, featureBuffer.length, 0.0f)
      util.Arrays.fill(labelBuffer, 0, labelBuffer.length, 0.0f)

      // One-Hot format for feature
      // Expected transformed format should be:
      //
      // Example1: Input = [0, 2, 3], label = [2, 3, 1], dictionary length = 4
      // Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      // Transformed: label = [3, 4, 2] (+1 because Tensor index starts from 1)
      //
      // Example2: Input = [0, 2, 3], label = [0], dictionary length = 4
      // Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      // Transformed: label = [1] (+1 because Tensor index starts from 1)

      var i = 0
      while (i < sentence.dataLength) {
        featureBuffer(i*vocabLength + sentence.getData(i).toInt) = 1.0f
        i += 1
      }
      while (i < dataLength) {
        val index = (RNG.uniform(0.0, 1.0) * vocabLength).toInt
        featureBuffer(i*vocabLength + index) = 1.0f
        i += 1
      }

      i = 0
      while (i < sentence.labelLength) {
        labelBuffer(i) = sentence.label()(i) + 1.0f
        i += 1
      }
      while (i < labelLength) {
        val index = (RNG.uniform(0.0, 1.0) * vocabLength).toInt
        labelBuffer(i) = index.toFloat + 1.0f
        i += 1
      }

      buffer.copy(featureBuffer, labelBuffer,
        Array(dataLength, vocabLength), Array(labelLength))
    })
  }
}
