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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.Iterator
import scala.reflect.ClassTag

object TextToLabeledSentence {
  def apply[T: ClassTag](dictionary: Dictionary)
           (implicit ev: TensorNumeric[T])
  : TextToLabeledSentence[T] =
    new TextToLabeledSentence[T](dictionary)
  def apply[T: ClassTag](numSteps: Int)(implicit ev: TensorNumeric[T])
  : TextToSentenceWithSteps[T] = new TextToSentenceWithSteps[T](numSteps)
}

/**
 * Transform a string of sentence to LabeledSentence.
 * e.g. ["I", "love", "Intel"] => [0, 1, 2]
 *      data:  [0, 1]
 *      label: [1, 2]
 *
 * The input Array[String] should be a tokenized sentence.
 * e.g. I love Intel => ["I", "love", "Intel"]
 * @param dictionary
 * @param ev
 * @tparam T
 */
class TextToLabeledSentence[T: ClassTag](dictionary: Dictionary)
  (implicit ev: TensorNumeric[T])
  extends Transformer[Array[String], LabeledSentence[T]] {
  private val buffer = new LabeledSentence[T]()

  override def apply(prev: Iterator[Array[String]]): Iterator[LabeledSentence[T]] = {
    prev.map(sentence => {
      val indexes = sentence.map(x =>
        ev.fromType[Int](dictionary.getIndex(x)))
      val nWords = indexes.length - 1
      val data = indexes.take(nWords)
      val label = indexes.drop(1)
      buffer.copy(data, label)
    })
  }
}

/**
 * Transform a sequence of integers to LabeledSentence.
 * e.g. input = [0, 1, 2, 3, 4, 5, 6, ..]
 *      numSteps = 3
 *
 *      xbuffer = [0, 1, 2]
 *      ybuffer = [1, 2, 3]
 *
 * next:
 *      xbuffer = [3, 4, 5]
 *      ybuffer = [4, 5, 6]
 * @param numSteps
 * @param ev$1
 * @param ev
 * @tparam T
 */
private[bigdl] class TextToSentenceWithSteps[T: ClassTag](numSteps: Int)
  (implicit ev: TensorNumeric[T])
  extends Transformer[Array[T], LabeledSentence[T]] {
  val xbuffer = new Array[T](numSteps)
  val ybuffer = new Array[T](numSteps)
  val buffer = new LabeledSentence[T]()

  override def apply(prev: Iterator[Array[T]]): Iterator[LabeledSentence[T]] = {
    prev.map(sentence => {
      require(sentence.length >= numSteps + 1,
        "input sentence length should be numSteps + 1, " +
          s"sentence.length = ${sentence.length}, numSteps = ${numSteps}")
      Array.copy(sentence, 0, xbuffer, 0, numSteps)
      Array.copy(sentence, 1, ybuffer, 0, numSteps)

      buffer.copy(xbuffer, ybuffer)
      buffer
    })
  }
}
