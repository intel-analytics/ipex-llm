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
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{FlatSpec, Matchers}

class FillerSepc extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "Filler all" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Filler(0, 0, 1, 1, 255) -> MatToFloats()
    val transformed = transformer(data)
    val imf = transformed.asInstanceOf[LocalImageFrame].array(0)
    imf.floats().forall(x => x == 255) should be (true)
  }

  "Filler part" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Filler(0, 0, 1, 0.5f, 255) -> MatToFloats()
    val transformed = transformer(data)
    val imf = transformed.asInstanceOf[LocalImageFrame].array(0)
    imf.floats().slice(0, 3 * 375 * 250).forall(_ == 255) should be (true)
  }

  "Filler part2" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Filler(0, 0, 1, 0.5f, 255)
    val transformed = transformer(data)

    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    val imf = transformed.asInstanceOf[LocalImageFrame].array(0)
    Imgcodecs.imwrite(tmpFile.toString, imf.opencvMat())
    println(tmpFile)
  }
}
