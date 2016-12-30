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

      require(dataLength == labelLength, "data length should be equal to label length")

      if (featureBuffer == null || featureBuffer.length < dataLength * vocabLength) {
        featureBuffer = new Array[Float](dataLength * vocabLength)
      }
      if (labelBuffer == null || labelBuffer.length < labelLength) {
        labelBuffer = new Array[Float](labelLength)
      }

      // One-Hot format for feature

      var i = 0
      while (i < sentence.dataLength()) {
        featureBuffer(i*vocabLength + sentence.getData(i).toInt) = 1.0f
        labelBuffer(i) = sentence.label()(i) + 1.0f
        i += 1
      }

      val lastIndex = labelBuffer(sentence.labelLength() - 1)
      while (i < dataLength) {
        val index = (RNG.uniform(0.0, 1.0) * vocabLength).toInt
        featureBuffer(i*vocabLength + index) = 1.0f
        labelBuffer(i-1) = index + 1.0f
        i += 1
      }
      labelBuffer(dataLength - 1) = lastIndex

      buffer.copy(featureBuffer, labelBuffer,
        Array(dataLength, vocabLength), Array(labelLength))
    })
  }
}
