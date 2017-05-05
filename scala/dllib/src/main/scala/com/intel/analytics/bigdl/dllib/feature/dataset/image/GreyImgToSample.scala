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

object GreyImgToSample {
  def apply(): GreyImgToSample = {
    new GreyImgToSample()
  }
}

/**
 * transform labeled grey image to sample
 */
class GreyImgToSample() extends Transformer[LabeledGreyImage, Sample[Float]] {

  private val buffer = Sample[Float]()
  private val labelBuffer = new Array[Float](1)
  private val featureSize = new Array[Int](2)
  private val labelSize = Array(1)

  override def apply(prev: Iterator[LabeledGreyImage]): Iterator[Sample[Float]] = {
    prev.map(img => {
      labelBuffer(0) = img.label()
      featureSize(0) = img.height()
      featureSize(1) = img.width()

      buffer.set(img.content, labelBuffer, featureSize, labelSize)
    })
  }
}
