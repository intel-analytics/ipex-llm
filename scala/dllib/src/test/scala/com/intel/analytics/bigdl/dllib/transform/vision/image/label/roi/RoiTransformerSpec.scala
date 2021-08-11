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

package com.intel.analytics.bigdl.transform.vision.image.label.roi

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, HFlip, Resize}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.{BboxUtil, BoundingBox}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.T
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{FlatSpec, Matchers}

class RoiTransformerSpec extends FlatSpec with Matchers {
  private def classes = Array(11.0, 11.0, 11.0, 16.0, 16.0, 16.0, 11.0, 16.0,
    16.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0).map(_.toFloat)

  private def visualize(mat: OpenCVMat, boxes: Tensor[Float]): Unit = {
    (1 to boxes.size(1)).foreach(i => {
      val bbox = BoundingBox(boxes.valueAt(i, 1), boxes.valueAt(i, 2),
        boxes.valueAt(i, 3), boxes.valueAt(i, 4))
      mat.drawBoundingBox(bbox, "")
    })
  }

  private def boxes = Array(2.0, 84.0, 59.0, 248.0,
    68.0, 115.0, 233.0, 279.0,
    64.0, 173.0, 377.0, 373.0,
    320.0, 2.0, 496.0, 375.0,
    221.0, 4.0, 341.0, 374.0,
    135.0, 14.0, 220.0, 148.0,
    69.0, 43.0, 156.0, 177.0,
    58.0, 54.0, 104.0, 139.0,
    279.0, 1.0, 331.0, 86.0,
    320.0, 22.0, 344.0, 96.0,
    337.0, 1.0, 390.0, 107.0).map(_.toFloat)

  private def getLables = {
    RoiLabel(Tensor(Storage(classes)).resize(2, 11),
      Tensor(Storage(boxes)).resize(11, 4))
  }

  val resource = getClass.getClassLoader.getResource("pascal/")

  "RoiNormalize" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = RoiNormalize()
    transformer(images)

    imageFeature.getLabel[RoiLabel].classes should be(label.classes)

    val expected = Tensor[Float](T(0.004, 0.22399999, 0.11800001, 0.6613333,
      0.136, 0.30666667, 0.46600002, 0.744,
      0.128, 0.46133333, 0.754, 0.99466664,
      0.64000005, 0.0053333333, 0.99200004, 1.0,
      0.44200003, 0.010666667, 0.68200004, 0.99733335,
      0.27, 0.037333332, 0.44000003, 0.39466667,
      0.13800001, 0.11466666, 0.312, 0.472,
      0.116000004, 0.144, 0.208, 0.37066665,
      0.558, 0.0026666666, 0.66200006, 0.22933333,
      0.64000005, 0.058666665, 0.688, 0.25599998,
      0.674, 0.0026666666, 0.78000003, 0.28533334)).resize(11, 4)

    imageFeature.getLabel[RoiLabel].bboxes should be(expected)
  }

  "RoiHFlip" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = HFlip() -> RoiHFlip(false)
    transformer(images)

    imageFeature.getLabel[RoiLabel].classes should be(label.classes)

    val boxes = imageFeature.getLabel[RoiLabel].bboxes
    val expected = Tensor[Float](T(441.0, 84.0, 498.0, 248.0,
      267.0, 115.0, 432.0, 279.0,
      123.0, 173.0, 436.0, 373.0,
      4.0, 2.0, 180.0, 375.0,
      159.0, 4.0, 279.0, 374.0,
      280.0, 14.0, 365.0, 148.0,
      344.0, 43.0, 431.0, 177.0,
      396.0, 54.0, 442.0, 139.0,
      169.0, 1.0, 221.0, 86.0,
      156.0, 22.0, 180.0, 96.0,
      110.0, 1.0, 163.0, 107.0)).resize(11, 4)
    boxes should be(expected)

    visualize(imageFeature.opencvMat(), expected)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }

  "RoiHFlip normalized" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = RoiNormalize() -> RoiHFlip()
    transformer(images)

    imageFeature.getLabel[RoiLabel].classes should be(label.classes)

    val boxes = imageFeature.getLabel[RoiLabel].bboxes
    val expected = Tensor[Float](T(0.88199997, 0.22399999, 0.996, 0.6613333,
      0.534, 0.30666667, 0.86399996, 0.744,
      0.24599999, 0.46133333, 0.872, 0.99466664,
      0.007999957, 0.0053333333, 0.35999995, 1.0,
      0.31799996, 0.010666667, 0.55799997, 0.99733335,
      0.55999994, 0.037333332, 0.73, 0.39466667,
      0.68799996, 0.11466666, 0.862, 0.472,
      0.792, 0.144, 0.884, 0.37066665,
      0.33799994, 0.0026666666, 0.44199997, 0.22933333,
      0.31199998, 0.058666665, 0.35999995, 0.25599998,
      0.21999997, 0.0026666666, 0.32599998, 0.28533334)).resize(11, 4)
    boxes should be(expected)
  }

  "RoiResize normalized" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile) -> Resize(300, 300)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = RoiNormalize() -> RoiResize()
    transformer(images)

    imageFeature.getLabel[RoiLabel].classes should be(label.classes)

    val boxes = imageFeature.getLabel[RoiLabel].bboxes
    val expected = Tensor[Float](T(0.004, 0.22399999, 0.11800001, 0.6613333,
      0.136, 0.30666667, 0.46600002, 0.744,
      0.128, 0.46133333, 0.754, 0.99466664,
      0.64000005, 0.0053333333, 0.99200004, 1.0,
      0.44200003, 0.010666667, 0.68200004, 0.99733335,
      0.27, 0.037333332, 0.44000003, 0.39466667,
      0.13800001, 0.11466666, 0.312, 0.472,
      0.116000004, 0.144, 0.208, 0.37066665,
      0.558, 0.0026666666, 0.66200006, 0.22933333,
      0.64000005, 0.058666665, 0.688, 0.25599998,
      0.674, 0.0026666666, 0.78000003, 0.28533334)).resize(11, 4)
    boxes should be(expected)
  }

  "RoiResize normalized == false" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile) -> Resize(300, 300)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = RoiResize()
    transformer(images)

    imageFeature.getLabel[RoiLabel].classes should be(label.classes)

    val boxes = imageFeature.getLabel[RoiLabel].bboxes
    val expected = Tensor[Float](T(1.2, 67.200005, 35.4, 198.40001,
      40.800003, 92.0, 139.8, 223.2,
      38.4, 138.40001, 226.20001, 298.4,
      192.0, 1.6, 297.6, 300.0,
      132.6, 3.2, 204.6, 299.2,
      81.0, 11.2, 132.0, 118.4,
      41.4, 34.4, 93.600006, 141.6,
      34.800003, 43.2, 62.4, 111.200005,
      167.40001, 0.8, 198.6, 68.8,
      192.0, 17.6, 206.40001, 76.8,
      202.20001, 0.8, 234.00002, 85.6)).resize(11, 4)
    boxes should be(expected)

    visualize(imageFeature.opencvMat(), expected)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }

  "RoiProject" should "work properly" in {
    val label = getLables
    val images = ImageFrame.read(resource.getFile)
    val imageFeature = images.asInstanceOf[LocalImageFrame].array(0)
    imageFeature(ImageFeature.label) = label

    val transformer = CenterCrop(300, 300) -> RoiNormalize() -> RoiProject()
    transformer(images)

    val boxes = imageFeature.getLabel[RoiLabel].bboxes
    val expected = Tensor[Float](T(0.0, 0.25833336, 0.44333336, 0.805,
      0.0, 0.45166665, 0.9233333, 1.0,
      0.40333334, 0.0, 0.8033333, 1.0,
      0.116666675, 0.0, 0.4, 0.36833334,
      0.0, 0.01833333, 0.1866667, 0.46500003,
      0.5966667, 0.0, 0.77, 0.16166666,
      0.73333335, 0.0, 0.8133333, 0.195,
      0.78999996, 0.0, 0.9666667, 0.23166668)).resize(8, 4)
    boxes should be(expected)

    BboxUtil.scaleBBox(expected, 300, 300)
    visualize(imageFeature.opencvMat(), expected)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
  }
}
