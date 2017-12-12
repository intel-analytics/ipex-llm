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

package com.intel.analytics.bigdl.transform.vision.image.augmentation

import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFrame, LocalImageFrame, MatToFloats}
import org.scalatest.{FlatSpec, Matchers}

class PixelNormalizerSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "PixelNormalizer" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val means = new Array[Float](375 * 500 * 3)
    var i = 0
    while (i < 375 * 500 * 3) {
      means(i) = 300f
      means(i + 1) = 200f
      means(i + 2) = 100f
      i += 3
    }
    val transformer = PixelNormalizer(means) -> MatToFloats()
    val transformed = transformer(data)

    val data2 = ImageFrame.read(resource.getFile)
    val toFloat = new MatToFloatsWithNorm(meanRGB = Some(100f, 200f, 300f))
    val transformed2 = toFloat(data2)

    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    val imageFeature2 = transformed2.asInstanceOf[LocalImageFrame].array(0)
    imageFeature2.floats().length should be (375 * 500 * 3)
    imageFeature2.floats() should equal(imageFeature.floats())
  }
}
