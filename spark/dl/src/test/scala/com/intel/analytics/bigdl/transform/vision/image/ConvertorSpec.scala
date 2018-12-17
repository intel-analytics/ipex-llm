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

package com.intel.analytics.bigdl.transform.vision.image

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.scalatest.{FlatSpec, Matchers}

class ConvertorSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "MatToFloat" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val imF = data.asInstanceOf[LocalImageFrame].array.head
    val float = OpenCVMat.toFloatPixels(imF.opencvMat())
    val data2 = ImageFrame.read(resource.getFile)
    val transformer = MatToFloats()
    transformer(data2)
    data2.toLocal().array(0).floats() should equal(float._1)
  }

  "MatToFloat no share" should "work properly" in {
    val resource = getClass.getClassLoader.getResource("imagenet/n02110063")
    val data = ImageFrame.read(resource.getFile)
    val transformer = MatToFloats(shareBuffer = false)
    transformer(data)
    val array = data.toLocal().array
    array(0).floats().equals(array(1).floats()) should be (false)
    array(0).floats().equals(array(2).floats()) should be (false)
    array(1).floats().equals(array(2).floats()) should be (false)
  }

  "MatToTensor" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val imF = data.asInstanceOf[LocalImageFrame].array.head
    val tensor2 = imF.toTensor(ImageFeature.floats)
    val transformer = MatToTensor[Float]()
    transformer(data)
    val tensor = imF[Tensor[Float]](ImageFeature.imageTensor)
    tensor should be (tensor2)
  }

  "MatToTensor" should "work when copy grey image to RGB" in {
    val resource = getClass.getClassLoader.getResource("grey")
    val data = ImageFrame.read(resource.getFile)
    val transformer = MatToTensor[Float](greyToRGB = true)
    transformer(data)
    val image = data.toLocal().array.head
    image.getSize should be ((500, 357, 1))
    val tensor = image[Tensor[Float]]("imageTensor")
    tensor.size(1) should be(3)
    tensor.size(2) should be(500)
    tensor.size(3) should be(357)
    tensor(1) should be(tensor(2))
    tensor(1) should be(tensor(3))
  }

  "toTensor" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    MatToFloats()(data)
    val imF = data.asInstanceOf[LocalImageFrame].array.head
    val tensor2 = imF.toTensor(ImageFeature.floats)

    val data2 = ImageFrame.read(resource.getFile)
    val transformer = MatToTensor[Float]()
    transformer(data2)
    val tensor = data2.toLocal().array.head[Tensor[Float]](ImageFeature.imageTensor)
    tensor should be (tensor2)
  }
}
