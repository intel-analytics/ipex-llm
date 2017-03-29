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
import com.intel.analytics.bigdl.tensor.Tensor

object BGRImgPixelNormalizer {
  def apply(means: Tensor[Float]): BGRImgPixelNormalizer
  = new BGRImgPixelNormalizer(means)
}

/**
 * Each pixel value of the input BGR Image sub the given mean value of the corresponding chanel
 * @param means mean value of BGR
 */
class BGRImgPixelNormalizer(means: Tensor[Float])
  extends Transformer[LabeledBGRImage, LabeledBGRImage] {

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      val content = img.content
      val meansData = means.storage().array()
      require(content.length % 3 == 0)
      require(content.length == means.nElement())
      var i = 0
      while (i < content.length) {
        content(i + 2) = content(i + 2) - meansData(i + 2)
        content(i + 1) = content(i + 1) - meansData(i + 1)
        content(i + 0) = content(i + 0) - meansData(i + 0)
        i += 3
      }
      img
    })
  }
}
