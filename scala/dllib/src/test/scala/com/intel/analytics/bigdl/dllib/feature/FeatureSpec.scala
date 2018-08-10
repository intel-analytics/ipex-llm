/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.common.{BigDLAdapter, Preprocessing}
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageResize, ImageSet}
import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class FeatureSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("imagenet/n04370456/")
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test Feature Engineering").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "BigDLAdapter" should "adapt BigDL Transformer" in {
    val newResize = BigDLAdapter(ImageResize(1, 1))
    assert(newResize.isInstanceOf[Preprocessing[_, _]])
  }

  "Local ImageSet" should "work with resize" in {
    val image = ImageSet.read(resource.getFile, resizeH = 200, resizeW = 200)
    val imf = image.toLocal().array.head
    require(imf.getHeight() == 200)
    require(imf.getWidth() == 200)
  }

  "Distribute ImageSet" should "work with resize" in {
    val image = ImageSet.read(resource.getFile, sc, resizeH = 200, resizeW = 200)
    val imf = image.toDistributed().rdd.collect().head
    require(imf.getHeight() == 200)
    require(imf.getWidth() == 200)
  }

  "Local ImageSet" should "work with bytes" in {
    val files = Utils.listLocalFiles(resource.getFile)
    val bytes = files.map { p =>
      FileUtils.readFileToByteArray(p)
    }
    ImageSet.array(bytes)
  }

  "Distribute ImageSet" should "work with bytes" in {
    val data = sc.binaryFiles(resource.getFile).map { case (p, stream) =>
      stream.toArray()
    }
    val images = ImageSet.rddBytes(data)
    images.rdd.collect()
  }

  "ImageBytesToMat" should "work with png and jpg" in {
    val path = getClass.getClassLoader.getResource("png").getFile
    val image = ImageSet.read(path, sc)
    val image2 = ImageSet.read(path, sc)
    val jpg = image -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
    val png = image2 -> ImageBytesToMat()
    val imfJpg = jpg.toDistributed().rdd.collect().head
    val imfPng = png.toDistributed().rdd.collect().head
    val (height, width, channel) = imfJpg.opencvMat().shape()
    val (height2, width2, channel2) = imfPng.opencvMat().shape()
    require(height == height2 && width == width2)
    require(channel == 3 && channel2 == 4)
  }
}
