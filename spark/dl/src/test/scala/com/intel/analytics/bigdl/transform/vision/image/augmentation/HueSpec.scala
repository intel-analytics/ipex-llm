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

import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFrame, LocalImageFrame}
import org.scalatest.{FlatSpec, Matchers}

class HueSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")
  "Hue" should "work properly" in {
    val data = ImageFrame.read(resource.getFile) -> BytesToMat()
    val transformer = Hue(-1, 1)
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]
    transformed.array(0).getHeight() should be (transformed.array(0).getOriginalHeight)
    transformed.array(0).getWidth() should be (transformed.array(0).getOriginalWidth)
  }
}
