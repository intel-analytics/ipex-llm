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

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFrame, LocalImageFrame}
import org.scalatest.{FlatSpec, Matchers}

class ResizeSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")
  "resize" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Resize(300, 300)
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]
    transformed.array(0).getHeight() should be(300)
    transformed.array(0).getWidth() should be(300)
  }

  "AspectScale" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = AspectScale(750, maxSize = 3000)
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]
    transformed.array(0).getHeight() should be(750)
    transformed.array(0).getWidth() should be(1000)
  }

  "RandomAspectScale" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = RandomAspectScale(Array(750), maxSize = 3000)
    val transformed = transformer(data).asInstanceOf[LocalImageFrame]
    transformed.array(0).getHeight() should be(750)
    transformed.array(0).getWidth() should be(1000)
  }

  "getWidthHeightAfterRatioScale" should "work" in {
    val img = OpenCVMat.read(resource.getFile + "/000025.jpg")
    val (height, width) = AspectScale.getHeightWidthAfterRatioScale(img, 600, 1000, 1)
    height should be (600)
    width should be (800)
  }
}
