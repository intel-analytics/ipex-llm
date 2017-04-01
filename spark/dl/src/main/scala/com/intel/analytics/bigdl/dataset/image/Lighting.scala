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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.collection.Iterator

object Lighting {
  def apply(): Lighting = {
    new Lighting()
  }
}

/**
 * Lighting noise for data augmentation
 *
 * Krizhevsky et al. proposed fancy PCA when training the famous Alex-Net in 2012
 * Fancy PCA alters the intensities of the RGB channels in training images
 * For each training image, add the corresponding quantity to each RGB image pixel
 */
class Lighting extends Transformer[LabeledBGRImage, LabeledBGRImage] {
  val alphastd = 0.1f
  val eigval = Tensor[Float](Storage(Array(0.2175f, 0.0188f, 0.0045f)), 1, Array(3))
  val eigvec = Tensor[Float](Storage(Array(-0.5675f, 0.7192f, 0.4009f,
    -0.5808f, -0.0045f, -0.8140f,
    -0.5836f, -0.6948f, 0.4203f)), 1, Array(3, 3))

  def lighting(input: Array[Float]): Unit = {
    if (alphastd != 0) {
      val alpha = Tensor[Float](3).apply1(_ => RNG.uniform(0, alphastd).toFloat)
      val rgb = eigvec.clone
        .cmul(alpha.view(1, 3).expand(Array(3, 3)))
        .cmul(eigval.view(1, 3).expand(Array(3, 3)))
        .sum(2).squeeze
      var i = 0
      while (i < input.length) {
        input(i) = input(i) + rgb.storage().array()(0)
        input(i + 1) = input(i + 1) + rgb.storage().array()(1)
        input(i + 2) = input(i + 2) + rgb.storage().array()(2)
        i += 3
      }
    }
  }

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      lighting(img.content)
      img
    })
  }
}
