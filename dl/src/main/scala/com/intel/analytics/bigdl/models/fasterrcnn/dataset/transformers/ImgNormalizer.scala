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

package com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.ImageWithRoi

object ImgNormalizer {
  def apply(mean: (Float, Float, Float)): ImgNormalizer = {
    new ImgNormalizer(mean._1, mean._2, mean._3)
  }
}

class ImgNormalizer(meanR: Float, meanG: Float, meanB: Float)
  extends Transformer[ImageWithRoi, ImageWithRoi] {

  override def apply(prev: Iterator[ImageWithRoi]): Iterator[ImageWithRoi] = {
    prev.map(img => transform(img))
  }

  def transform(img: ImageWithRoi): ImageWithRoi = {
    val content = img.data
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = content(i + 2) - meanR
      content(i + 1) = content(i + 1) - meanG
      content(i + 0) = content(i + 0) - meanB
      i += 3
    }
    img
  }
}
