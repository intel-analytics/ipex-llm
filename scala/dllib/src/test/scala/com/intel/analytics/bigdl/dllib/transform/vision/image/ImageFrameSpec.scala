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

import java.io.File
import java.nio.file.Paths

import com.google.common.io.Files
import com.intel.analytics.bigdl.dataset.image.{BGRImage, BGRImgToLocalSeqFile, BytesToGreyImg, GreyImgNormalizer, GreyImgToSample, LocalImgReaderWithName, HFlip => BHFlip}
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.utils.{Engine, TestUtils}
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ImageFrameSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("pascal/")
  var sc: SparkContext = null
  var sqlContext: SQLContext = null
  before {
    val conf = Engine.createSparkConf().setAppName("ImageSpec").setMaster("local[2]")
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
    Engine.init
  }

  after {
    if (null != sc) sc.stop()
  }

  "read LocalImageFrame" should "work properly" in {
    val local = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    local.array.length should be(1)
    val imf = local.array(0)
    assert(imf.uri.endsWith("000025.jpg"))
    assert(imf.bytes.length == 95959)
    imf.opencvMat().shape() should be((375, 500, 3))
  }

  "LocalImageFrame toDistributed" should "work properly" in {
    val local = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    local.array.foreach(x => println(x.uri, x.bytes.length))
    val imageFeature = local.toDistributed(sc).rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.opencvMat().shape() should be((375, 500, 3))
  }

  "read DistributedImageFrame" should "work properly" in {
    val distributed = ImageFrame.read(resource.getFile, sc)
      .asInstanceOf[DistributedImageFrame]
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.opencvMat().shape() should be((375, 500, 3))
  }

  "read DistributedImageFrame with partition number" should "work properly" in {
    val tmpFile = Files.createTempDir()
    val dir = new File(tmpFile.toString + "/images")
    dir.mkdir()
    (1 to 10).foreach(i => {
      Files.copy(new File(resource.getFile + "000025.jpg"), new File(dir + s"/$i.jpg"))
    })

    val distributed = ImageFrame.read(dir.toString, sc, 5)
      .asInstanceOf[DistributedImageFrame]
    if (tmpFile.exists()) FileUtils.deleteDirectory(tmpFile)
  }

  "SequenceFile write and read" should "work properly" in {
    val tmpFile = Files.createTempDir()
    val dir = tmpFile.toString + "/parque"
    ImageFrame.writeParquet(resource.getFile, dir, sqlContext, 1)

    val distributed = ImageFrame.readParquet(dir, sqlContext)
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    FileUtils.deleteDirectory(tmpFile)
  }

  "read local" should "work" in {
    val images = ImageFrame.read(resource.getFile).asInstanceOf[LocalImageFrame]
    images.array.length should be (1)

    val images2 = ImageFrame.read(resource.getFile + "*.jpg").asInstanceOf[LocalImageFrame]
    images2.array.length should be (1)

    val images3 = ImageFrame.read(resource.getFile + "000025.jpg").asInstanceOf[LocalImageFrame]
    images3.array.length should be (1)

    val images4 = ImageFrame.read(resource.getFile + "0000251.jpg").asInstanceOf[LocalImageFrame]
    images4.array.length should be (0)
  }

  "transform" should "work" in {
    val transformer = BytesToMat() -> HFlip()
    val images = ImageFrame.read(resource.getFile)
    images.transform(transformer)
  }


  "ImageNet" should "load" in {
    // generate seq for test
    TestUtils.cancelOnWindows()
    val resource = getClass().getClassLoader().getResource("imagenet")
    val tmpFile = java.io.File.createTempFile("UnitTest", System.nanoTime().toString)
    require(tmpFile.delete())
    require(tmpFile.mkdir())

    // Convert the test imagenet files to seq files
    val files = (DataSet.ImageFolder.paths(Paths.get(processPath(resource.getPath())))
      -> LocalImgReaderWithName(BGRImage.NO_SCALE)
      -> BGRImgToLocalSeqFile(2, Paths.get(tmpFile.getAbsolutePath(), "imagenet"))
      ).toLocal().data(train = false).map(s => {
      println(s);
      s
    }).toArray

    val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(tmpFile.getAbsolutePath(), sc, 10) ->
      PixelBytesToMat() ->
      Resize(256, 256) ->
      RandomCrop(224, 224) ->
      RandomTransformer(HFlip(), 0.5) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor[Float](toRGB = true) ->
      ImageFrameToSample[Float](Array(ImageFeature.imageTensor), Array(ImageFeature.label))

    val sampleRdd = imageFrame.toDistributed().rdd.map(x => x[Sample[Float]](ImageFeature.sample))
    sampleRdd.foreach(x => {
      require(x.feature().size(1) == 3)
      require(x.feature().size(2) == 224)
      require(x.feature().size(3) == 224)
      println(x.label())
    })

    if (tmpFile.exists()) tmpFile.delete()
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "mnist data source" should "load image with ImageFrame correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSet = com.intel.analytics.bigdl.models.lenet.Utils.load(
      processPath(resource.getPath()) + File.separator + "t10k-images.idx3-ubyte",
      processPath(resource.getPath()) + File.separator + "t10k-labels.idx1-ubyte")
    val array = dataSet.map(x => {
      val im = ImageFeature(x.data, x.label)
      im(ImageFeature.originalSize) = (28, 28, 1)
      im
    })
    val testMean = 0
    val testStd = 1
    val imf = ImageFrame.array(array)
    val transformer = PixelBytesToMat() ->
      ChannelNormalize(testMean * 255, testStd * 255) ->
      ChannelNormalize(testMean, testStd) ->
      MatToTensor[Float](shareBuffer = false) ->
      ImageFrameToSample[Float]()
    val transformed = transformer(imf)
    transformed.toLocal().array.foreach(x => {
      println(x(ImageFeature.sample))
    })

    val transformer2 =
      BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToSample()
    val evaluationSet = DataSet.array(dataSet).transform(transformer2)
    evaluationSet.toLocal().data(false).toArray.zip(transformed.toLocal().array).foreach(x => {
      x._1.feature() should be (x._2[Sample[Float]](ImageFeature.sample).feature())
    })
  }

  "read gray scale image" should "work" in {
    val resource = getClass().getClassLoader().getResource("gray/gray.bmp")
    val imf = ImageFrame.read(resource.getFile)
    imf.toLocal().array(0).getOriginalSize should be (50, 50, 1)
  }

  "transform gray scale image" should "work" in {
    val resource = getClass().getClassLoader().getResource("gray/gray.bmp")
    val imf = ImageFrame.read(resource.getFile) -> Resize(28, 28) -> MatToTensor[Float]()
    imf.toLocal().array(0).getOriginalSize should be (50, 50, 1)
    imf.toLocal().array(0).getSize should be (28, 28, 1)
  }
}
