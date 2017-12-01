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

import com.intel.analytics.bigdl.transform.vision.image.{ImageFrame, LocalImageFrame, MatToFloats}
import org.scalatest.{FlatSpec, Matchers}

class ChannelNormalizeSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "ChannelNormalize" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = ChannelNormalize(100, 200, 300) -> MatToFloats()
    val transformed = transformer(data)
    val imf = transformed.asInstanceOf[LocalImageFrame].array(0)

    val toFloat = MatToFloats(meanRGB = Some(100f, 200f, 300f))
    val data2 = ImageFrame.read(resource.getFile)
    val transformed2 = toFloat(data2)
    val imf2 = transformed2.asInstanceOf[LocalImageFrame].array(0)
    imf2.floats().length should be (375 * 500 * 3)
    imf2.floats() should equal(imf.floats())
  }

  "ChannelNormalize with std not 1" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = ChannelNormalize(100, 200, 300, 2, 2, 2) -> MatToFloats()
    val transformed = transformer(data)
    val imf = transformed.asInstanceOf[LocalImageFrame].array(0)

    val data2 = ImageFrame.read(resource.getFile)
    val toFloat = MatToFloats(meanRGB = Some(100f, 200f, 300f))
    val transformed2 = toFloat(data2)
    val imf2 = transformed2.asInstanceOf[LocalImageFrame].array(0)

    imf2.floats().length should be (375 * 500 * 3)
    imf2.floats().map(_ / 2) should equal(imf.floats())
  }
}
