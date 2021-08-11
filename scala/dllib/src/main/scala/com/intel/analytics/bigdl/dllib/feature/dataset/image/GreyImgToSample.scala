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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

object GreyImgToSample {
  def apply(): GreyImgToSample = {
    new GreyImgToSample()
  }
}

/**
 * transform labeled grey image to sample
 */
class GreyImgToSample() extends Transformer[LabeledGreyImage, Sample[Float]] {

  private val featureBuffer = Tensor[Float]()
  private val labelBuffer = Tensor[Float](1)
  private val featureSize = new Array[Int](2)

  override def apply(prev: Iterator[LabeledGreyImage]): Iterator[Sample[Float]] = {
    prev.map(img => {
      labelBuffer.storage.array()(0) = img.label()
      featureSize(0) = img.height()
      featureSize(1) = img.width()
      featureBuffer.set(Storage(img.content), sizes = featureSize)

      Sample(featureBuffer, labelBuffer)
    })
  }
}
