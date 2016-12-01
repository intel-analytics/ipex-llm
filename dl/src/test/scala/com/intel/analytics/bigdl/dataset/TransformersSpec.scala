/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.{FlatSpec, Matchers}

class TransformersSpec extends FlatSpec with Matchers {
  import Utils._

  "Grey Image Cropper" should "crop image correct" in {
    val image = new LabeledImage(32, 32, 1)
    val tensor = Tensor[Float](Storage[Float](image.content), 1, Array(32, 32))
    tensor.rand()
    RNG.setSeed(1000)
    val cropper = new ImageCropper(24, 24, 1)
    val iter = cropper.transform(Iterator.single(image))
    val result = iter.next()

    result.width should be(24)
    result.width should be(24)

    val originContent = image.content
    val resultContent = result.content
    var y = 0
    while (y < 24) {
      var x = 0
      while (x < 24) {
        resultContent(y * 24 + x) should be(originContent((y + 1) * 32 + x + 5))
        x += 1
      }
      y += 1
    }
  }

  "Grey Image Normalizer" should "normalize image correctly" in {
    val image1 = new LabeledImage((1 to 9).map(_.toFloat).toArray, 3, 3, 1, 0)
    val image2 = new LabeledImage((10 to 18).map(_.toFloat).toArray, 3, 3, 1, 0)
    val image3 = new LabeledImage((19 to 27).map(_.toFloat).toArray, 3, 3, 1, 0)

    val mean = (1 to 27).sum.toDouble / 27
    val std = math.sqrt((1 to 27).map(e => (e - mean) * (e - mean)).sum / 27f)
    val target = image1.content.map(e => (e - mean) / std)

    val dataSource = new ArrayDataSource[LabeledImage](looped = false) {
      override protected val data: Array[LabeledImage] = Array(image1, image2, image3) }

    val normalizer = new ImageNormalizer(dataSource)
    val iter = normalizer.transform(Iterator.single(image1))
    val test = iter.next()
    normalizer.getMean(0) should be(mean)
    normalizer.getStd(0) should be(std)

    test.content.zip(target).foreach { case (a, b) => a should be(b.toFloat) }
  }

  "Grey Image toTensor" should "convert correctly" in {
    val image1 = new LabeledImage(32, 32, 1)
    val image2 = new LabeledImage(32, 32, 1)
    val image3 = new LabeledImage(32, 32, 1)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val dataSource = new ArrayDataSource[LabeledImage](true) {
      override protected val data: Array[LabeledImage] = Array(image1, image2, image3)
    }

    val toTensor = new ImageToTensor(2)
    val tensorDataSource = dataSource -> toTensor
    val (tensorResult1, labelTensor1) = tensorDataSource.next()
    tensorResult1.size(1) should be(2)
    tensorResult1.size(2) should be(32)
    tensorResult1.size(3) should be(32)
    val testData1 = tensorResult1.storage().array()
    val content1 = image1.content
    var i = 0
    while (i < content1.length) {
      testData1(i) should be(content1(i))
      i += 1
    }
    val content2 = image2.content
    i = 0
    while (i < content2.length) {
      testData1(i + 32 * 32) should be(content2(i))
      i += 1
    }
    val (tensorResult2, labelTensor2) = tensorDataSource.next()
    val content3 = image3.content
    tensorResult2.size(1) should be(2)
    tensorResult2.size(2) should be(32)
    tensorResult2.size(3) should be(32)
    i = 0
    while (i < content3.length) {
      testData1(i) should be(content3(i))
      i += 1
    }
    i = 0
    while (i < content1.length) {
      testData1(i + 32 * 32) should be(content1(i))
      i += 1
    }
  }

  "RGB Image Cropper" should "crop image correct" in {
    val image = new LabeledImage(32, 32, 3)
    val tensor = Tensor[Float](Storage[Float](image.content), 1, Array(3, 32, 32))
    tensor.rand()
    RNG.setSeed(1000)
    val cropper = new ImageCropper(24, 24, 3)
    val iter = cropper.transform(Iterator.single(image))
    val result = iter.next()

    result.width should be(24)
    result.width should be(24)

    val originContent = image.content
    val resultContent = result.content
    var c = 0
    while (c < 3) {
      var y = 0
      while (y < 24) {
        var x = 0
        while (x < 24) {
          resultContent((y * 24 + x) * 3 + c) should be(originContent((37 + y * 32 + x) * 3 +
            c))
          x += 1
        }
        y += 1
      }
      c += 1
    }
  }

  "RGB Image Normalizer" should "normalize image correctly" in {
    val image1 = new LabeledImage((1 to 27).map(_.toFloat).toArray, 3, 3, 3, 0)
    val image2 = new LabeledImage((2 to 28).map(_.toFloat).toArray, 3, 3, 3, 0)
    val image3 = new LabeledImage((3 to 29).map(_.toFloat).toArray, 3, 3, 3, 0)

    val firstFrameMean = (1 to 27).sum.toFloat / 27
    val firstFrameStd = math.sqrt((1 to 27).map(e => (e - firstFrameMean) * (e - firstFrameMean))
      .sum / 27).toFloat
    val secondFrameMean = (2 to 28).sum.toFloat / 27
    val secondFrameStd = math.sqrt((2 to 28).map(e => (e - secondFrameMean) * (e - secondFrameMean))
      .sum / 27).toFloat
    val thirdFrameMean = (3 to 29).sum.toFloat / 27
    val thirdFrameStd = math.sqrt((3 to 29).map(e => (e - thirdFrameMean) * (e - thirdFrameMean))
      .sum / 27).toFloat

    var i = 0
    val target = image1.content.map(e => {
      val r = if (i % 3 == 0) {
        (e - firstFrameMean) / firstFrameStd
      } else if (i % 3 == 1) {
        (e - secondFrameMean) / secondFrameStd
      } else {
        (e - thirdFrameMean) / thirdFrameStd
      }
      i += 1
      r
    })

    val dataSource = new ArrayDataSource[LabeledImage](false) {
      override protected val data: Array[LabeledImage] = Array(image1, image2, image3)
    }

    val normalizer = new ImageNormalizer(dataSource)
    val iter = normalizer.transform(Iterator.single(image1))
    val test = iter.next()
    normalizer.getMean should be(Array(firstFrameMean, secondFrameMean, thirdFrameMean))
    val stds = normalizer.getStd
    stds(0) should be(firstFrameStd.toDouble +- 1e-6)
    stds(1) should be(secondFrameStd.toDouble +- 1e-6)
    stds(2) should be(thirdFrameStd.toDouble +- 1e-6)

    test.content.zip(target).foreach { case (a, b) => a should be(b +- 1e-6f) }
  }

  "RGB Image toTensor" should "convert correctly" in {
    val image1 = new LabeledImage(32, 32, 3)
    val image2 = new LabeledImage(32, 32, 3)
    val image3 = new LabeledImage(32, 32, 3)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(3, 32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(3, 32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(3, 32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val dataSource = new ArrayDataSource[LabeledImage](true) {
      override protected val data: Array[LabeledImage] = Array(image1, image2, image3)
    }

    val toTensor = new ImageToTensor(2)
    val tensorDataSource = dataSource -> toTensor
    val (tensorResult1, labelTensor1) = tensorDataSource.next()
    tensorResult1.size(1) should be(2)
    tensorResult1.size(2) should be(3)
    tensorResult1.size(3) should be(32)
    tensorResult1.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    tensorResult1.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    tensorResult1.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    val (tensorResult2, labelTensor2) = tensorDataSource.next()
    val content3 = image3.content
    tensorResult2.size(1) should be(2)
    tensorResult2.size(2) should be(3)
    tensorResult2.size(3) should be(32)
    tensorResult2.size(4) should be(32)

    i = 0
    tensorResult2.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })
    i = 0
    tensorResult2.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })
  }

  "Multi thread RGB Image toTensor" should "convert correctly" in {
    val image1 = new LabeledImage(32, 32, 3)
    val image2 = new LabeledImage(32, 32, 3)
    val image3 = new LabeledImage(32, 32, 3)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(3, 32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(3, 32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(3, 32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val dataSource = new ArrayDataSource[LabeledImage](true) {
      override protected val data: Array[LabeledImage] = Array(image1, image2, image3)
    }

    val toTensor = new MultiThreadImageToSingleTensor[LabeledImage](
      width = 32, height = 32, numChannels = 3,
      threadNum = 2, batchSize = 2, transformer = Identity[LabeledImage]()
    )
    val tensorDataSource = dataSource -> toTensor
    val (tensorResult1, labelTensor1) = tensorDataSource.next()
    tensorResult1.size(1) should be(2)
    tensorResult1.size(2) should be(3)
    tensorResult1.size(3) should be(32)
    tensorResult1.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    tensorResult1.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    tensorResult1.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult1.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    val (tensorResult2, labelTensor2) = tensorDataSource.next()
    val content3 = image3.content
    tensorResult2.size(1) should be(2)
    tensorResult2.size(2) should be(3)
    tensorResult2.size(3) should be(32)
    tensorResult2.size(4) should be(32)

    i = 0
    tensorResult2.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })
    i = 0
    tensorResult2.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    tensorResult2.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })
  }

  "Image To SeqFile" should "be good" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val pathToImage = PathToRGBImage(ImageUtils.NO_SCALE)
    val dataSource = new ImageNetDataSource(Paths.get(processPath(resource.getPath())), looped =
      false)

    RandomGenerator.RNG.setSeed(1000)

    dataSource.shuffle()
    val tmpFile = Paths.get(java.io.File.createTempFile("UnitTest", "ImageToSeqFile").getPath)
    val seqWriter = new ImageToSequentialFile(2, tmpFile)
    val writePipeline = dataSource -> pathToImage -> seqWriter
    while(writePipeline.hasNext) {
      println(s"writer file ${writePipeline.next()}")
    }

    val seqDataSource = new ArrayDataSource[Path](false) {
      override protected val data: Array[Path] = Array(
        Paths.get(tmpFile + "_0"),
        Paths.get(tmpFile + "_1"),
        Paths.get(tmpFile + "_2"),
        Paths.get(tmpFile + "_3"),
        Paths.get(tmpFile + "_4"),
        Paths.get(tmpFile + "_5")
      )
    }
    dataSource.reset()
    var count = 0
    val readPipeline = seqDataSource -> SeqFileToArrayByte() -> ArrayByteToRGBImage()
    readPipeline.zip(dataSource -> pathToImage).foreach { case (l, r) =>
      l.label should be(r.label)
      l.width should be(r.width)
      l.height should be(r.height)
      l.content.zip(r.content).foreach(d => d._1 should be(d._2))
      count += 1
    }

    count should be(11)
  }

  "Image Scale" should "be good" in {
    val resource = getClass.getClassLoader.getResource("imagenet/n02110063/n02110063_8651.JPEG")

    val pipeline = PathToRGBImage(ImageUtils.NO_SCALE) + new Scaler(10, 10) + new ImageToTensor(1)
    val result = pipeline
      .transform(Iterator.single((1f, Paths.get(resource.toURI))))
      .next()._1
      .squeeze(1)
      .select(1, 3)
      .apply1(x => (math.round(x * 1e4) / 1e4).toFloat)

    val expectedResult = Tensor[Float](Storage[Float](Array(
      0.5062, 0.6190, 0.6759, 0.7176, 0.6846, 0.7291, 0.6203, 0.5416, 0.5553, 0.7444,
      0.4792, 0.5666, 0.6313, 0.6368, 0.6376, 0.6409, 0.6625, 0.6457, 0.5728, 0.6074,
      0.4225, 0.4470, 0.4751, 0.5300, 0.4738, 0.5751, 0.5106, 0.4560, 0.4431, 0.5500,
      0.3956, 0.4207, 0.4562, 0.4363, 0.5239, 0.4287, 0.4321, 0.4251, 0.3823, 0.5441,
      0.3220, 0.3916, 0.4142, 0.4951, 0.5896, 0.3535, 0.4421, 0.3874, 0.3854, 0.5378,
      0.3219, 0.3241, 0.3761, 0.5862, 0.7083, 0.3330, 0.4273, 0.3212, 0.3133, 0.5264,
      0.3387, 0.3886, 0.3442, 0.3904, 0.4745, 0.3621, 0.3009, 0.3430, 0.3038, 0.5246,
      0.2814, 0.3035, 0.2811, 0.3044, 0.3591, 0.3408, 0.3287, 0.3289, 0.2626, 0.5330,
      0.2196, 0.2654, 0.3217, 0.3371, 0.2965, 0.3224, 0.3523, 0.3654, 0.3171, 0.5165,
      0.7287, 0.5968, 0.5986, 0.5924, 0.5957, 0.5988, 0.6000, 0.5914, 0.5844, 0.7317
    ).map(x => x.toFloat)),
      1, Array(10, 10))


    expectedResult should equal (result)
  }
}
