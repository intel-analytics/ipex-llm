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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.RandomGenerator

import scala.collection.Iterator

object HFlip {
  def apply(threshold: Double = 0.0): HFlip = {
    new HFlip(threshold)
  }
}

/**
 * Flip a image with a probability. The threshold higher, the less easier to flip the image.
 * @param threshold
 */
class HFlip(threshold: Double) extends Transformer[LabeledBGRImage, LabeledBGRImage] {
  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      if (RandomGenerator.RNG.uniform(0, 1) >= threshold) {
        img.hflip()
      } else {
        img
      }
    })
  }
}

