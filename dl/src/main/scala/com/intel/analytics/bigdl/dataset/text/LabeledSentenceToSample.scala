/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

object LabeledSentenceToSample {
  def apply(vocabLength: Int): LabeledSentenceToSample =
    new LabeledSentenceToSample(vocabLength)
}

class LabeledSentenceToSample(vocabLength: Int)
  extends Transformer[LabeledSentence, Sample] {
  private val buffer = new Sample()
  private var arrayBuffer: Array[Float] = null
  private var labelBuffer: Array[Float] = null

  override def apply(prev: Iterator[LabeledSentence]): Iterator[Sample] = {
    prev.map(other => {
      if (arrayBuffer == null || arrayBuffer.length < other.length * vocabLength) {
        arrayBuffer = new Array[Float](other.length*vocabLength)
      }
      var i = 0
      while (i < other.length) {
        arrayBuffer(i*vocabLength + other.getData(i).toInt) = 1.0f
        i += 1
      }
      if (labelBuffer == null || labelBuffer.length < other.length) {
        labelBuffer = new Array[Float](other.length)
      }
      i = 0
      while (i < other.length) {
        labelBuffer(i) = other.label()(i) + 1.0f
      }
      buffer.copy(arrayBuffer, labelBuffer,
        Array(other.length, vocabLength), Array(other.labelLength))
    })
  }
}
