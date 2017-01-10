/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.{FlatSpec, Matchers}

class TransformersSpec extends FlatSpec with Matchers {
  import TestUtils._

  "Grey Image Cropper" should "crop image correct" in {
    val image = new LabeledGreyImage(32, 32)
    val tensor = Tensor[Float](Storage[Float](image.content), 1, Array(32, 32))
    tensor.rand()
    RNG.setSeed(1000)
    val cropper = new GreyImgCropper(24, 24)
    val iter = cropper.apply(Iterator.single(image))
    val result = iter.next()

    result.width() should be(24)
    result.height() should be(24)

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
    val image1 = new LabeledGreyImage((1 to 9).map(_.toFloat).toArray, 3, 3, 0)
    val image2 = new LabeledGreyImage((10 to 18).map(_.toFloat).toArray, 3, 3, 0)
    val image3 = new LabeledGreyImage((19 to 27).map(_.toFloat).toArray, 3, 3, 0)

    val mean = (1 to 27).sum.toFloat / 27
    val std = math.sqrt((1 to 27).map(e => (e - mean) * (e - mean)).sum / 27f).toFloat
    val target = image1.content.map(e => (e - mean) / std)

    val dataSet = new LocalArrayDataSet[LabeledGreyImage](
      Array(image1, image2, image3))

    val normalizer = GreyImgNormalizer(dataSet)
    val iter = normalizer.apply(Iterator.single(image1))
    val test = iter.next()
    normalizer.getMean() should be(mean)
    normalizer.getStd() should be(std)

    test.content.zip(target).foreach { case (a, b) => a should be(b) }
  }

  "Grey Image toTensor" should "convert correctly" in {
    Engine.setNodeNumber(None)
    val image1 = new LabeledGreyImage(32, 32)
    val image2 = new LabeledGreyImage(32, 32)
    val image3 = new LabeledGreyImage(32, 32)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val dataSet = new LocalArrayDataSet[LabeledGreyImage](Array(image1, image2, image3))

    val toTensor = new GreyImgToBatch(2)
    val tensorDataSet = dataSet -> toTensor
    val iter = tensorDataSet.toLocal().data(train = true)
    val batch = iter.next()
    batch.data.size(1) should be(2)
    batch.data.size(2) should be(32)
    batch.data.size(3) should be(32)
    val testData1 = batch.data.storage().array()
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
    val batch2 = iter.next()
    val content3 = image3.content
    batch2.data.size(1) should be(2)
    batch2.data.size(2) should be(32)
    batch2.data.size(3) should be(32)
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
    val image = new LabeledBGRImage(32, 32)
    val tensor = Tensor[Float](Storage[Float](image.content), 1, Array(3, 32, 32))
    tensor.rand()
    RNG.setSeed(1000)
    val cropper = new BGRImgCropper(24, 24)
    val iter = cropper.apply(Iterator.single(image))
    val result = iter.next()

    result.width() should be(24)
    result.height() should be(24)

    val originContent = image.content
    val resultContent = result.content
    var c = 0
    while (c < 3) {
      var y = 0
      while (y < 24) {
        var x = 0
        while (x < 24) {
          resultContent((y * 24 + x) * 3 + c) should be(originContent(582 + (y * 32 + x) * 3 +
            c))
          x += 1
        }
        y += 1
      }
      c += 1
    }
  }

  "RGB image cropper with padding" should "crop correctly" in {
    val (w, h, channel, label, cW, cH, pad) = (2, 1, 3, 1, 4, 3, 1)
    val image = new LabeledBGRImage(Array[Float](1, 2, 3, 4, 5, 6), w, h, label)
    val padAndCropData = Array[Float](
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )

    val cropper = new BGRImgRdmCropper(cropHeight = cH, cropWidth = cW, padding = pad)
    val iter = cropper.apply(Iterator.single(image))
    val result = iter.next()

    result.width() should be(cW)
    result.height() should be(cH)
    result.content should be(padAndCropData)
  }

  "RGB Image Normalizer" should "normalize image correctly" in {
    val image1 = new LabeledBGRImage((1 to 27).map(_.toFloat).toArray, 3, 3, 0)
    val image2 = new LabeledBGRImage((2 to 28).map(_.toFloat).toArray, 3, 3, 0)
    val image3 = new LabeledBGRImage((3 to 29).map(_.toFloat).toArray, 3, 3, 0)

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

    val dataSet = new LocalArrayDataSet[LabeledBGRImage](Array(image1, image2, image3))

    val normalizer = BGRImgNormalizer(dataSet)
    val iter = normalizer.apply(Iterator.single(image1))
    val test = iter.next()
    normalizer.getMean() should be((thirdFrameMean, secondFrameMean, firstFrameMean))
    val stds = normalizer.getStd()
    stds._1 should be(firstFrameStd.toDouble +- 1e-6)
    stds._2 should be(secondFrameStd.toDouble +- 1e-6)
    stds._3 should be(thirdFrameStd.toDouble +- 1e-6)

    test.content.zip(target).foreach { case (a, b) => a should be(b +- 1e-6f) }
  }

  "RGB Image toTensor" should "convert correctly" in {
    Engine.setNodeNumber(None)
    val image1 = new LabeledBGRImage(32, 32)
    val image2 = new LabeledBGRImage(32, 32)
    val image3 = new LabeledBGRImage(32, 32)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(3, 32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(3, 32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(3, 32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val dataSet = new LocalArrayDataSet[LabeledBGRImage](Array(image1, image2, image3))

    val toTensor = new BGRImgToBatch(2)
    val tensorDataSet = dataSet -> toTensor
    val iter = tensorDataSet.toLocal().data(train = true)
    val batch1 = iter.next()
    batch1.data.size(1) should be(2)
    batch1.data.size(2) should be(3)
    batch1.data.size(3) should be(32)
    batch1.data.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    batch1.data.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch1.data.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch1.data.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    batch1.data.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch1.data.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch1.data.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    val batch = iter.next()
    val content3 = image3.content
    batch.data.size(1) should be(2)
    batch.data.size(2) should be(3)
    batch.data.size(3) should be(32)
    batch.data.size(4) should be(32)

    i = 0
    batch.data.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })
    i = 0
    batch.data.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
  }

  "Multi thread RGB Image toTensor" should "convert correctly" in {
    Engine.setNodeNumber(None)
    val image1 = new LabeledBGRImage(32, 32)
    val image2 = new LabeledBGRImage(32, 32)
    val image3 = new LabeledBGRImage(32, 32)
    val tensor1 = Tensor[Float](Storage[Float](image1.content), 1, Array(3, 32, 32))
    val tensor2 = Tensor[Float](Storage[Float](image2.content), 1, Array(3, 32, 32))
    val tensor3 = Tensor[Float](Storage[Float](image3.content), 1, Array(3, 32, 32))
    tensor1.rand()
    tensor2.rand()
    tensor3.rand()

    val core = Engine.coreNumber()
    Engine.setCoreNumber(1)
    val dataSet = new LocalArrayDataSet[LabeledBGRImage](Array(image1, image2, image3))
    val toTensor = new MTLabeledBGRImgToBatch[LabeledBGRImage](
      width = 32, height = 32, totalBatchSize = 2, transformer = Identity[LabeledBGRImage]
    )
    val tensorDataSet = dataSet -> toTensor
    val iter = tensorDataSet.toLocal().data(train = true)
    val batch = iter.next()
    batch.data.size(1) should be(2)
    batch.data.size(2) should be(3)
    batch.data.size(3) should be(32)
    batch.data.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    batch.data.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    batch.data.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch.data.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    val batch2 = iter.next()
    val content3 = image3.content
    batch2.data.size(1) should be(2)
    batch2.data.size(2) should be(3)
    batch2.data.size(3) should be(32)
    batch2.data.size(4) should be(32)

    i = 0
    batch2.data.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch2.data.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch2.data.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })
    i = 0
    batch2.data.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    batch2.data.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    batch2.data.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    Engine.setCoreNumber(core)
  }

  "RGBImage To SeqFile" should "be good" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val pathToImage = LocalImgReader(BGRImage.NO_SCALE)
    val dataSet = DataSet.ImageFolder.paths(
      Paths.get(processPath(resource.getPath()))
    )

    RandomGenerator.RNG.setSeed(1000)

    dataSet.shuffle()
    val tmpFile = Paths.get(java.io.File.createTempFile("UnitTest", "RGBImageToSeqFile").getPath)
    val seqWriter = BGRImgToLocalSeqFile(2, tmpFile)
    val writePipeline = dataSet -> pathToImage -> seqWriter
    val iter = writePipeline.toLocal().data(train = false)
    while (iter.hasNext) {
      println(s"writer file ${iter.next()}")
    }

    val seqDataSet = new LocalArrayDataSet[LocalSeqFilePath](Array(
      LocalSeqFilePath(Paths.get(tmpFile + "_0.seq")),
      LocalSeqFilePath(Paths.get(tmpFile + "_1.seq")),
      LocalSeqFilePath(Paths.get(tmpFile + "_2.seq")),
      LocalSeqFilePath(Paths.get(tmpFile + "_3.seq")),
      LocalSeqFilePath(Paths.get(tmpFile + "_4.seq")),
      LocalSeqFilePath(Paths.get(tmpFile + "_5.seq"))
    ))
    var count = 0
    val readPipeline = seqDataSet -> LocalSeqFileToBytes() -> BytesToBGRImg()
    val readIter = readPipeline.toLocal().data(train = false)
    readIter.zip((dataSet -> pathToImage).toLocal().data(train = false))
      .foreach { case (l, r) =>
      l.label() should be(r.label())
      l.width() should be(r.width())
      l.height() should be(r.height())
      l.content.zip(r.content).foreach(d => d._1 should be(d._2))
      count += 1
    }

    count should be(11)
  }
}
