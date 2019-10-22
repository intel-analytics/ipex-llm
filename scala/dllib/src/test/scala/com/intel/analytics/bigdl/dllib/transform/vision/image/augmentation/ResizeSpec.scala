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
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, ImageFrame, LocalImageFrame}
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{FlatSpec, Matchers}

class ResizeSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")
  "resize" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Resize(300, 300)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(300)
    imageFeature.getWidth() should be(300)

    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }

  "resize useScaleFactor false" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Resize(300, 300, useScaleFactor = false)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(300)
    imageFeature.getWidth() should be(300)

    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }

  "AspectScale" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = AspectScale(750, maxSize = 3000)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(750)
    imageFeature.getWidth() should be(1000)


    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }

  "RandomAspectScale" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = RandomAspectScale(Array(750), maxSize = 3000)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(750)
    imageFeature.getWidth() should be(1000)
  }

  "getWidthHeightAfterRatioScale" should "work" in {
    val img = OpenCVMat.read(resource.getFile + "/000025.jpg")
    val (height, width) = AspectScale.getHeightWidthAfterRatioScale(img, 600, 1000, 1)
    height should be (600)
    width should be (800)
  }

  "scaleResize without roi" should "be ok" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = ScaleResize(minSize = 100, maxSize = 120, resizeROI = false)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be(90)
    imageFeature.getWidth() should be(120)
  }
}
