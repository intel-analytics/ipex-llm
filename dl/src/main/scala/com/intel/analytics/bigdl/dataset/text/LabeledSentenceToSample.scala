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

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}

import scala.collection.Iterator
import java.util

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object LabeledSentenceToSample {
  def apply[T: ClassTag]
  (vocabLength: Int,
   fixDataLength: Option[Int] = None,
   fixLabelLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T])
  : LabeledSentenceToSample[T] =
    new LabeledSentenceToSample[T](
      vocabLength,
      fixDataLength,
      fixLabelLength)
}

class LabeledSentenceToSample[T: ClassTag](vocabLength: Int,
                              fixDataLength: Option[Int],
                              fixLabelLength: Option[Int])
  (implicit ev: TensorNumeric[T])
  extends Transformer[LabeledSentence[T], Sample[T]] {
  private val buffer = Sample[T]()
  private var featureBuffer: Array[T] = null
  private var labelBuffer: Array[T] = null

  override def apply(prev: Iterator[LabeledSentence[T]]): Iterator[Sample[T]] = {
    prev.map(sentence => {

      val dataLength = fixDataLength.getOrElse(sentence.dataLength())
      val labelLength = fixLabelLength.getOrElse(sentence.labelLength())

      if (featureBuffer == null || featureBuffer.length < dataLength * vocabLength) {
        featureBuffer = new Array[T](dataLength * vocabLength)
      }
      if (labelBuffer == null || labelBuffer.length < labelLength) {
        labelBuffer = new Array[T](labelLength)
      }

      // initialize featureBuffer to 0.0

      ev.getType() match {
        case DoubleType =>
          util.Arrays.fill(featureBuffer.asInstanceOf[Array[Double]], 0, featureBuffer.length, 0.0)
          util.Arrays.fill(labelBuffer.asInstanceOf[Array[Double]], 0, labelBuffer.length, 0.0)
        case FloatType =>
          util.Arrays.fill(featureBuffer.asInstanceOf[Array[Float]], 0, featureBuffer.length, 0.0f)
          util.Arrays.fill(labelBuffer.asInstanceOf[Array[Float]], 0, labelBuffer.length, 0.0f)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }

      /* One-Hot format for feature
       * Expected transformed format should be:
       *
       * Example1: Input = [0, 2, 3], label = [2, 3, 1], dictionary length = 4
       * Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
       * Transformed: label = [3, 4, 2] (+1 because Tensor index starts from 1)
       *
       * Example2: Input = [0, 2, 3], label = [0], dictionary length = 4
       * Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
       * Transformed: label = [1] (+1 because Tensor index starts from 1)
       */


      val startTokenIndex = sentence.getData(0)
      val endTokenIndex = if (labelLength == 1) 0
        else ev.toType[Int](sentence.getLabel(sentence.labelLength - 1))

      var i = 0
      while (i < sentence.dataLength) {
        featureBuffer(i*vocabLength + ev.toType[Int](sentence.getData(i)))
          = ev.fromType[Float](1.0f)
        i += 1
      }
      while (i < dataLength) {
        featureBuffer(i*vocabLength + endTokenIndex) = ev.fromType[Float](1.0f)
        i += 1
      }

      i = 0
      while (i < sentence.labelLength) {
        labelBuffer(i) = ev.plus(sentence.label()(i), ev.fromType[Float](1.0f))
        i += 1
      }
      while (i < labelLength) {
        labelBuffer(i) = ev.plus(startTokenIndex, ev.fromType[Float](1.0f))
        i += 1
      }

      buffer.set(featureBuffer, labelBuffer,
        Array(dataLength, vocabLength), Array(labelLength))
    })
  }
}
