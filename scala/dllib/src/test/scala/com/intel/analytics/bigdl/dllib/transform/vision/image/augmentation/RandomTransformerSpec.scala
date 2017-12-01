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

class RandomTransformerSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "RandomTransformer with 0" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = RandomTransformer(FixedCrop(0, 0, 50, 50, false), 0)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(375)
    imageFeature.getWidth() should be(500)
  }

  "RandomTransformer with 1" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = RandomTransformer(FixedCrop(0, 0, 50, 50, false), 1)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(50)
    imageFeature.getWidth() should be(50)
  }
}
