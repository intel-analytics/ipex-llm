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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.Iterator
import scala.reflect.ClassTag

object BGRImgToSample {
  def apply(toRGB: Boolean = true): BGRImgToSample = {
    new BGRImgToSample(toRGB)
  }
}

/**
 * transform labeled bgr image to sample
 */
class BGRImgToSample(toRGB: Boolean = true) extends Transformer[LabeledBGRImage, Sample[Float]] {

  private val buffer = Sample[Float]()
  private val labelBuffer = new Array[Float](1)
  private var featureBuffer: Array[Float] = null
  private val featureSize = new Array[Int](3)
  private val labelSize = Array(1)

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[Sample[Float]] = {
    featureSize(0) = 3
    prev.map(img => {
      labelBuffer(0) = img.label()
      if (featureBuffer == null || featureBuffer.length < (3 * img.height() * img.width())) {
        featureBuffer = new Array[Float](img.height() * img.width() * 3)
        featureSize(1) = img.height()
        featureSize(2) = img.width()
      }

      img.copyTo(featureBuffer, 0, toRGB)
      buffer.set(featureBuffer, labelBuffer, featureSize, labelSize)
    })
  }
}
