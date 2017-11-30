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

class ChannelNormalizeSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "ChannelNormalize" should "work properly" in {
    val data = ImageFrame.read(resource.getFile) -> BytesToMat()
    val transformer = ChannelNormalize(100, 200, 300) -> MatToFloats()
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]

    val toFloat = MatToFloats(meanRGB = Some(100f, 200f, 300f))
    val data2 = ImageFrame.read(resource.getFile) -> BytesToMat()
    val transformed2 = toFloat(data2).asInstanceOf[LocalImageFrame]
    transformed2.array(0).floats().length should be (375 * 500 * 3)
    transformed2.array(0).floats() should equal(transformed.array(0).floats())
  }

  "ChannelNormalize with std not 1" should "work properly" in {
    val data = ImageFrame.read(resource.getFile) -> BytesToMat()
    val transformer = ChannelNormalize(100, 200, 300, 2, 2, 2) -> MatToFloats()
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]

    val data2 = ImageFrame.read(resource.getFile) -> BytesToMat()
    val toFloat = MatToFloats(meanRGB = Some(100f, 200f, 300f))
    val transformed2 = toFloat(data2).asInstanceOf[LocalImageFrame]

    transformed2.array(0).floats().length should be (375 * 500 * 3)
    transformed2.array(0).floats().map(_ / 2) should equal(transformed.array(0).floats())
  }
}
