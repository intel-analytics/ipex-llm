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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image
import com.intel.analytics.bigdl.transform.vision.image.MatToFloats._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.common.{BigDLAdapter, Preprocessing}
import com.intel.analytics.zoo.feature.image._
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.reflect.ClassTag


class FeatureSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("imagenet/n04370456/")
  val imagenet = getClass.getClassLoader.getResource("imagenet")
  val gray = getClass.getClassLoader.getResource("gray")
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
    require(imf.getChannel() == 3)

    val imageGray = ImageSet.read(gray.getFile, resizeH = 200, resizeW = 200)
    val imfGray = imageGray.toLocal().array.head
    require(imfGray.getHeight() == 200)
    require(imfGray.getWidth() == 200)
    require(imfGray.getChannel() == 1)
  }

  "Distribute ImageSet" should "work with resize" in {
    val image = ImageSet.read(resource.getFile, sc, resizeH = 200, resizeW = 200)
    val imf = image.toDistributed().rdd.collect().head
    require(imf.getHeight() == 200)
    require(imf.getWidth() == 200)
    require(imf.getChannel() == 3)

    val imageGray = ImageSet.read(gray.getFile, sc, resizeH = 200, resizeW = 200)
    val imfGray = imageGray.toDistributed().rdd.collect().head
    require(imfGray.getHeight() == 200)
    require(imfGray.getWidth() == 200)
    require(imfGray.getChannel() == 1)
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
    images.toDistributed().rdd.collect()
  }

  private def getRelativePath(path: String) = {
    val fileUri = new java.io.File(path).toURI
    val currentUri = new java.io.File("").toURI
    currentUri.relativize(fileUri).getRawPath
  }

  "Distributed ImageSet" should "read relative path" in {
    val relativePath = getRelativePath(imagenet.getFile)
    val image = ImageSet.read(relativePath, sc, withLabel = true)
    image.toDistributed().rdd.collect()
  }

  "Local ImageSet" should "read relative path" in {
    val relativePath = getRelativePath(imagenet.getFile)
    val image = ImageSet.read(relativePath, withLabel = true)
    image.toLocal().array
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

  "ImageNormalize" should "work with min max normType" in {
    val image = ImageSet.read(resource.getFile, sc)
    val jpg = image -> PerImageNormalize(0, 1) -> ImageMatToFloats()

    val imfJpg = jpg.toDistributed().rdd.collect().head
    imfJpg.floats().foreach{ t => assert(t>=0 && t<=1.0)}
  }

  "ImageMatToTensor" should "work with both NCHW and NHWC" in {
    val resource = getClass.getClassLoader.getResource("pascal/")
    val data = ImageSet.read(resource.getFile)
    val nhwc = (data -> ImageMatToTensor[Float](format = DataFormat.NHWC)).toLocal()
      .array.head.apply[Tensor[Float]](ImageFeature.imageTensor)
    require(nhwc.isContiguous() == true)

    val data2 = ImageSet.read(resource.getFile)
    require(data2.toLocal().array.head.apply[Tensor[Float]](ImageFeature.imageTensor) == null)
    val nchw = (data2 -> ImageMatToTensor[Float]()).toLocal()
      .array.head.apply[Tensor[Float]](ImageFeature.imageTensor)

    require(nchw.transpose(1, 2).transpose(2, 3).contiguous().storage().array().deep
      == nhwc.storage().array().deep)
  }

  "ImageChannelNormalize with std not 1" should "work properly" in {
    val data = ImageSet.read(resource.getFile)
    val transformer = ImageChannelNormalize(100, 200, 300, 4, 3, 2) -> ImageMatToTensor[Float]()
    val transformed = transformer(data)
    val imf = transformed.asInstanceOf[LocalImageSet].array(0)

    val data2 = ImageFrame.read(resource.getFile)
    val toFloat = new MatToFloatsWithNorm(meanRGB = Some(100f, 200f, 300f),
      stdRGB = Some(4f, 3f, 2f))
    val transformed2 = toFloat(data2)
    val imf2 = transformed2.asInstanceOf[LocalImageFrame].array(0)

    imf2.floats().length should be (270 * 290 * 3)
    imf2.floats() should equal(imf.floats())
  }
}

class MatToFloatsWithNorm(validHeight: Int = 300, validWidth: Int = 300, validChannels: Int = 3,
  meanRGB: Option[(Float, Float, Float)] = None,
  stdRGB: Option[(Float, Float, Float)] = None, outKey: String = ImageFeature.floats)
  extends FeatureTransformer {
  @transient private var data: Array[Float] = _

  private def normalize(img: Array[Float],
    meanR: Float, meanG: Float, meanB: Float,
    stdR: Float = 1f, stdG: Float = 1f, stdB: Float = 1f): Array[Float] = {
    val content = img
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = (content(i + 2) - meanR) / stdR
      content(i + 1) = (content(i + 1) - meanG) / stdG
      content(i + 0) = (content(i + 0) - meanB) / stdB
      i += 3
    }
    img
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    var input: OpenCVMat = null
    val (height, width, channel) = if (feature.isValid) {
      input = feature.opencvMat()
      (input.height(), input.width(), input.channels())
    } else {
      (validHeight, validWidth, validChannels)
    }
    if (null == data || data.length < height * width * channel) {
      data = new Array[Float](height * width * channel)
    }
    if (feature.isValid) {
      try {
        OpenCVMat.toFloatPixels(input, data)
        normalize(data, meanRGB.get._1, meanRGB.get._2, meanRGB.get._3
          , stdRGB.get._1, stdRGB.get._2, stdRGB.get._3)
      } finally {
        if (null != input) input.release()
        feature(ImageFeature.mat) = null
      }
    }
    feature(outKey) = data
    feature(ImageFeature.size) = (height, width, channel)
    feature
  }
}
