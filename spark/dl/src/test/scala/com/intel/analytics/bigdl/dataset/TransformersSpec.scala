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

package com.intel.analytics.bigdl.dataset

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.text.{LabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, TestUtils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{SequenceFile, Text}
import org.apache.hadoop.io.SequenceFile.Reader
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TransformersSpec extends FlatSpec with Matchers with BeforeAndAfter {
  import com.intel.analytics.bigdl.utils.TestUtils._

  before {
    Engine.setNodeAndCore(1, 1)
  }

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
    Engine.setNodeAndCore(1, 1)
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
    val input = batch.getInput().toTensor[Float]
    input.size(1) should be(2)
    input.size(2) should be(32)
    input.size(3) should be(32)
    val testData1 = input.storage().array()
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
    val input2 = batch.getInput().toTensor[Float]
    val content3 = image3.content
    input2.size(1) should be(2)
    input2.size(2) should be(32)
    input2.size(3) should be(32)
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
    Engine.setNodeAndCore(1, 1)
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
    val input1 = batch1.getInput().toTensor[Float]
    input1.size(1) should be(2)
    input1.size(2) should be(3)
    input1.size(3) should be(32)
    input1.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    input1.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input1.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input1.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    input1.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input1.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input1.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    val batch = iter.next()
    val input = batch.getInput().toTensor[Float]
    val content3 = image3.content
    input.size(1) should be(2)
    input.size(2) should be(3)
    input.size(3) should be(32)
    input.size(4) should be(32)

    i = 0
    input.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })
    i = 0
    input.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
  }

  "Multi thread RGB Image toTensor" should "convert correctly" in {
    Engine.setNodeNumber(1)
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
    val input = batch.getInput().toTensor[Float]
    input.size(1) should be(2)
    input.size(2) should be(3)
    input.size(3) should be(32)
    input.size(4) should be(32)
    val content1 = image1.content
    var i = 0
    input.select(1, 1).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input.select(1, 1).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input.select(1, 1).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    val content2 = image2.content
    i = 0
    input.select(1, 2).select(1, 1).apply1(e => {
      e should be(content2(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input.select(1, 2).select(1, 2).apply1(e => {
      e should be(content2(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input.select(1, 2).select(1, 3).apply1(e => {
      e should be(content2(i * 3))
      i += 1
      e
    })

    val batch2 = iter.next()
    val input2 = batch.getInput().toTensor[Float]
    val content3 = image3.content
    input2.size(1) should be(2)
    input2.size(2) should be(3)
    input2.size(3) should be(32)
    input2.size(4) should be(32)

    i = 0
    input2.select(1, 1).select(1, 1).apply1(e => {
      e should be(content3(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input2.select(1, 1).select(1, 2).apply1(e => {
      e should be(content3(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input2.select(1, 1).select(1, 3).apply1(e => {
      e should be(content3(i * 3))
      i += 1
      e
    })
    i = 0
    input2.select(1, 2).select(1, 1).apply1(e => {
      e should be(content1(i * 3 + 2))
      i += 1
      e
    })

    i = 0
    input2.select(1, 2).select(1, 2).apply1(e => {
      e should be(content1(i * 3 + 1))
      i += 1
      e
    })

    i = 0
    input2.select(1, 2).select(1, 3).apply1(e => {
      e should be(content1(i * 3))
      i += 1
      e
    })
    Engine.setCoreNumber(core)
  }

  "RGBImage To SeqFile without file name" should "be good" in {
    TestUtils.cancelOnWindows()
    val resource = getClass().getClassLoader().getResource("imagenet")
    val pathToImage = LocalImgReaderWithName(BGRImage.NO_SCALE)
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
      l.label() should be(r._1.label())
      l.width() should be(r._1.width())
      l.height() should be(r._1.height())
      l.content.zip(r._1.content).foreach(d => d._1 should be(d._2))
      count += 1
    }

    count should be(11)
  }

  "RGBImage To SeqFile with file name" should "be good" in {
    TestUtils.cancelOnWindows()
    val resource = getClass().getClassLoader().getResource("imagenet")
    val pathToImage = LocalImgReaderWithName(BGRImage.NO_SCALE)
    val dataSet = DataSet.ImageFolder.paths(
      Paths.get(processPath(resource.getPath()))
    )

    RandomGenerator.RNG.setSeed(1000)

    dataSet.shuffle()
    val tmpFile = Paths.get(java.io.File.createTempFile("UnitTest", "RGBImageToSeqFile").getPath)
    val seqWriter = BGRImgToLocalSeqFile(2, tmpFile, true)
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

    val seqData = seqDataSet.toLocal().data(train = false)
    val data = dataSet.toLocal().data(train = false)

    var reader: SequenceFile.Reader = null
    val key = new Text()
    val value = new Text()
    var count = 0

    while (seqData.hasNext) {
      val l = seqData.next()
      if ((reader == null) || (!reader.next(key, value))) {
        reader = new SequenceFile.Reader(new Configuration,
          Reader.file(new Path(l.path.toAbsolutePath.toString)))
      }

      while (reader.next(key, value) && data.hasNext) {
        val r = data.next()
        val imgData = BGRImage.readImage(r.path, BGRImage.NO_SCALE)
        SeqFileFolder.readName(key).toString should be (r.path.getFileName.toString)
        SeqFileFolder.readLabel(key).toFloat should be (r.label)
        value.copyBytes() should be (imgData)
        count += 1
      }
    }
    count should be(11)
  }

  "LabeledSentence toSample" should "transform correctly for single label" in {
    val input1 = Array(1.0f, 2.0f, 3.0f)
    val target1 = Array(1.0f)
    val input2 = Array(2.0f, 1.0f, 0.0f, 4.0f)
    val target2 = Array(0.0f)
    val input3 = Array(0.0f, 4.0f)
    val target3 = Array(1.0f)
    val labeledSentence1 = new LabeledSentence[Float](input1, target1)
    val labeledSentence2 = new LabeledSentence[Float](input2, target2)
    val labeledSentence3 = new LabeledSentence[Float](input3, target3)

    val dataSet = new LocalArrayDataSet[LabeledSentence[Float]](Array(labeledSentence1,
      labeledSentence2, labeledSentence3))

    val labeledSentenceToSample = LabeledSentenceToSample[Float](5)
    val sampleDataSet = dataSet -> labeledSentenceToSample
    val iter = sampleDataSet.toLocal().data(train = false)

    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)), 1, Array(4, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1)), 1, Array(2, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(2.0f)), 1, Array(1))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(1.0f)), 1, Array(1))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(2.0f)), 1, Array(1))


    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val batch1 = iter.next()
    batch1.feature() should be (tensorInput1)
    batch1.label() should be (tensorTarget1)

    val batch2 = iter.next()
    batch2.feature should be (tensorInput2)
    batch2.label should be (tensorTarget2)

    val batch3 = iter.next()
    batch3.feature should be (tensorInput3)
    batch3.label should be (tensorTarget3)
  }
  "LabeledSentence toSample" should "transform correctly for single label Double" in {
    val input1 = Array(1.0, 2.0, 3.0)
    val target1 = Array(1.0)
    val input2 = Array(2.0, 1.0, 0.0, 4.0)
    val target2 = Array(0.0)
    val input3 = Array(0.0, 4.0)
    val target3 = Array(1.0)
    val labeledSentence1 = new LabeledSentence[Double](input1, target1)
    val labeledSentence2 = new LabeledSentence[Double](input2, target2)
    val labeledSentence3 = new LabeledSentence[Double](input3, target3)

    val dataSet = new LocalArrayDataSet[LabeledSentence[Double]](Array(labeledSentence1,
      labeledSentence2, labeledSentence3))

    val labeledSentenceToSample = LabeledSentenceToSample[Double](5)
    val sampleDataSet = dataSet -> labeledSentenceToSample
    val iter = sampleDataSet.toLocal().data(train = false)

    val tensorInput1 = Tensor[Double](Storage(
      Array(0.0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Double](Storage(
      Array(0.0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)), 1, Array(4, 5))
    val tensorInput3 = Tensor[Double](Storage(
      Array(1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1)), 1, Array(2, 5))
    val tensorTarget1 = Tensor[Double](Storage(
      Array(2.0)), 1, Array(1))
    val tensorTarget2 = Tensor[Double](Storage(
      Array(1.0)), 1, Array(1))
    val tensorTarget3 = Tensor[Double](Storage(
      Array(2.0)), 1, Array(1))


    val sample1 = Sample[Double](tensorInput1, tensorTarget1)
    val sample2 = Sample[Double](tensorInput2, tensorTarget2)
    val sample3 = Sample[Double](tensorInput3, tensorTarget3)

    val batch1 = iter.next()
    batch1.feature should be (tensorInput1)
    batch1.label should be (tensorTarget1)

    val batch2 = iter.next()
    batch2.feature should be (tensorInput2)
    batch2.label should be (tensorTarget2)

    val batch3 = iter.next()
    batch3.feature should be (tensorInput3)
    batch3.label should be (tensorTarget3)
  }
  "LabeledSentence toSample" should "transform correctly for padding sentences single label" in {
    val input1 = Array(1.0f, 2.0f, 3.0f)
    val target1 = Array(1.0f)
    val input2 = Array(2.0f, 1.0f, 0.0f, 4.0f)
    val target2 = Array(0.0f)
    val input3 = Array(0.0f, 4.0f)
    val target3 = Array(1.0f)
    val labeledSentence1 = new LabeledSentence[Float](input1, target1)
    val labeledSentence2 = new LabeledSentence[Float](input2, target2)
    val labeledSentence3 = new LabeledSentence[Float](input3, target3)

    val dataSet = new LocalArrayDataSet[LabeledSentence[Float]](Array(labeledSentence1,
      labeledSentence2, labeledSentence3))

    val labeledSentenceToSample = LabeledSentenceToSample[Float](5, fixDataLength = Option(4))
    val sampleDataSet = dataSet -> labeledSentenceToSample
    val iter = sampleDataSet.toLocal().data(train = false)

    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0)), 1, Array(4, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)), 1, Array(4, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(4, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(2.0f)), 1, Array(1))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(1.0f)), 1, Array(1))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(2.0f)), 1, Array(1))


    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val batch1 = iter.next()
    batch1.feature should be (tensorInput1)
    batch1.label should be (tensorTarget1)

    val batch2 = iter.next()
    batch2.feature should be (tensorInput2)
    batch2.label should be (tensorTarget2)

    val batch3 = iter.next()
    batch3.feature should be (tensorInput3)
    batch3.label should be (tensorTarget3)
  }
  "LabeledSentence toSample" should "transform correctly for language model label" in {
    val input1 = Array(0.0f, 2.0f, 3.0f)
    val target1 = Array(2.0f, 3.0f, 4.0f)
    val input2 = Array(0.0f, 1.0f, 0.0f, 2.0f)
    val target2 = Array(1.0f, 0.0f, 2.0f, 4.0f)
    val input3 = Array(0.0f, 3.0f)
    val target3 = Array(3.0f, 4.0f)
    val labeledSentence1 = new LabeledSentence[Float](input1, target1)
    val labeledSentence2 = new LabeledSentence[Float](input2, target2)
    val labeledSentence3 = new LabeledSentence[Float](input3, target3)

    val dataSet = new LocalArrayDataSet[LabeledSentence[Float]](Array(labeledSentence1,
      labeledSentence2, labeledSentence3))

    val labeledSentenceToSample = LabeledSentenceToSample[Float](5)
    val sampleDataSet = dataSet -> labeledSentenceToSample
    val iter = sampleDataSet.toLocal().data(train = false)

    val tensorInput1 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)), 1, Array(4, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 1, 0)), 1, Array(2, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4.0f, 5.0f)), 1, Array(3))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1.0f, 3.0f, 5.0f)), 1, Array(4))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(4.0f, 5.0f)), 1, Array(2))


    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val batch1 = iter.next()
    batch1.feature should be (tensorInput1)
    batch1.label should be (tensorTarget1)

    val batch2 = iter.next()
    batch2.feature should be (tensorInput2)
    batch2.label should be (tensorTarget2)

    val batch3 = iter.next()
    batch3.feature should be (tensorInput3)
    batch3.label should be (tensorTarget3)
  }
  "LabeledSentence toSample" should "transform correctly" +
    " for language model label padding sentences" in {
    val input1 = Array(0.0f, 2.0f, 3.0f)
    val target1 = Array(2.0f, 3.0f, 4.0f)
    val input2 = Array(0.0f, 1.0f, 0.0f, 2.0f)
    val target2 = Array(1.0f, 0.0f, 2.0f, 4.0f)
    val input3 = Array(0.0f, 3.0f)
    val target3 = Array(3.0f, 4.0f)
    val labeledSentence1 = new LabeledSentence[Float](input1, target1)
    val labeledSentence2 = new LabeledSentence[Float](input2, target2)
    val labeledSentence3 = new LabeledSentence[Float](input3, target3)

    val dataSet = new LocalArrayDataSet[LabeledSentence[Float]](Array(labeledSentence1,
      labeledSentence2, labeledSentence3))

    val labeledSentenceToSample = LabeledSentenceToSample[Float](5, Option(4), Option(4))
    val sampleDataSet = dataSet -> labeledSentenceToSample
    val iter = sampleDataSet.toLocal().data(train = false)

    val tensorInput1 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1)), 1, Array(4, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)), 1, Array(4, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(4, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4.0f, 5.0f, 1.0f)), 1, Array(4))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1.0f, 3.0f, 5.0f)), 1, Array(4))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(4.0f, 5.0f, 1.0f, 1.0f)), 1, Array(4))


    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val batch1 = iter.next()
    batch1.feature should be (tensorInput1)
    batch1.label should be (tensorTarget1)

    val batch2 = iter.next()
    batch2.feature should be (tensorInput2)
    batch2.label should be (tensorTarget2)

    val batch3 = iter.next()
    batch3.feature should be (tensorInput3)
    batch3.label should be (tensorTarget3)
  }
  "SampleToBatchSpec" should "be good with TensorBatch1 Double" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Double](Storage(
      Array(0.0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Double](Storage(
      Array(0.0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Double](Storage(
      Array(1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Double](Storage(
      Array(3.0, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Double](Storage(
      Array(2.0, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Double](Storage(
      Array(5.0, 2, 1)), 1, Array(3))
    val sample1 = Sample[Double](tensorInput1, tensorTarget1)
    val sample2 = Sample[Double](tensorInput2, tensorTarget2)
    val sample3 = Sample[Double](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Double]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToBatch[Double](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = false)

    val batch1 = iter.next()

    val batch1Data = Tensor[Double](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Double](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Double](Array(1, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    val batch2Label = Tensor[Double](Array(1, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }
  "SampleToBatchSpec" should "be good with TensorBatch1" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(5.0f, 2, 1)), 1, Array(3))
    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Float]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToBatch[Float](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = false)

    val batch1 = iter.next()

    val batch1Data = Tensor[Float](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Float](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Float](Array(1, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    val batch2Label = Tensor[Float](Array(1, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }
  "SampleToBatchSpec" should "be good with TensorBatch2" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(5.0f, 2, 1)), 1, Array(3))
    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Float]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToBatch[Float](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = true)

    val batch1 = iter.next()

    val batch1Data = Tensor[Float](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Float](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Float](Array(2, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    batch2Data(2).resizeAs(tensorInput1).copy(tensorInput1)
    val batch2Label = Tensor[Float](Array(2, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2Label(2).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }

  "SampleToMiniBatchSpec" should "be good with TensorBatch1 Double" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Double](Storage(
      Array(0.0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Double](Storage(
      Array(0.0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Double](Storage(
      Array(1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Double](Storage(
      Array(3.0, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Double](Storage(
      Array(2.0, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Double](Storage(
      Array(5.0, 2, 1)), 1, Array(3))
    val sample1 = Sample[Double](tensorInput1, tensorTarget1)
    val sample2 = Sample[Double](tensorInput2, tensorTarget2)
    val sample3 = Sample[Double](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Double]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToMiniBatch[Double](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = false)

    val batch1 = iter.next()

    val batch1Data = Tensor[Double](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Double](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Double](Array(1, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    val batch2Label = Tensor[Double](Array(1, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }
  "SampleToMiniBatchSpec" should "be good with TensorBatch1" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(5.0f, 2, 1)), 1, Array(3))
    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Float]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToMiniBatch[Float](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = false)

    val batch1 = iter.next()

    val batch1Data = Tensor[Float](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Float](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Float](Array(1, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    val batch2Label = Tensor[Float](Array(1, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }
  "SampleToMiniBatchSpec" should "be good with TensorBatch2" in {
    Engine.setNodeAndCore(1, 1)
    val tensorInput1 = Tensor[Float](Storage(
      Array(0.0f, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), 1, Array(3, 5))
    val tensorInput2 = Tensor[Float](Storage(
      Array(0.0f, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)), 1, Array(3, 5))
    val tensorInput3 = Tensor[Float](Storage(
      Array(1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)), 1, Array(3, 5))
    val tensorTarget1 = Tensor[Float](Storage(
      Array(3.0f, 4, 5)), 1, Array(3))
    val tensorTarget2 = Tensor[Float](Storage(
      Array(2.0f, 1, 5)), 1, Array(3))
    val tensorTarget3 = Tensor[Float](Storage(
      Array(5.0f, 2, 1)), 1, Array(3))
    val sample1 = Sample[Float](tensorInput1, tensorTarget1)
    val sample2 = Sample[Float](tensorInput2, tensorTarget2)
    val sample3 = Sample[Float](tensorInput3, tensorTarget3)

    val dataSet = new LocalArrayDataSet[Sample[Float]](Array(sample1,
      sample2, sample3))
    val sampleToBatch = SampleToMiniBatch[Float](2)
    val sampleDataSet = dataSet -> sampleToBatch
    val iter = sampleDataSet.toLocal().data(train = true)

    val batch1 = iter.next()

    val batch1Data = Tensor[Float](Array(2, 3, 5))
    batch1Data(1).resizeAs(tensorInput1).copy(tensorInput1)
    batch1Data(2).resizeAs(tensorInput2).copy(tensorInput2)
    val batch1Label = Tensor[Float](Array(2, 3))
    batch1Label(1).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch1Label(2).resizeAs(tensorTarget2).copy(tensorTarget2)
    batch1.getInput should be (batch1Data)
    batch1.getTarget should be (batch1Label)

    val batch2 = iter.next()
    val batch2Data = Tensor[Float](Array(2, 3, 5))
    batch2Data(1).resizeAs(tensorInput3).copy(tensorInput3)
    batch2Data(2).resizeAs(tensorInput1).copy(tensorInput1)
    val batch2Label = Tensor[Float](Array(2, 3))
    batch2Label(1).resizeAs(tensorTarget3).copy(tensorTarget3)
    batch2Label(2).resizeAs(tensorTarget1).copy(tensorTarget1)
    batch2.getInput should be (batch2Data)
    batch2.getTarget should be (batch2Label)
  }

  "BRGImgToSample" should "be correct" in {
    RNG.setSeed(100)
    var data = new Array[Float](3 * 32 * 32)
    data = for (d <- data) yield RNG.uniform(1.0, 2.0).toFloat
    val image1 = new LabeledBGRImage(data, 32, 32, 1.0f)
    data = for (d <- data) yield RNG.uniform(1.1, 2.1).toFloat
    val image2 = new LabeledBGRImage(data, 32, 32, 2.0f)
    data = for (d <- data) yield RNG.uniform(1.2, 2.2).toFloat
    val image3 = new LabeledBGRImage(data, 32, 32, 3.0f)

    val image = Array(image1, image2, image3)
    val toSample = BGRImgToSample() -> SampleToMiniBatch(1)
    val miniBatch1 = toSample(image.toIterator)
    val miniBatch2 = BGRImgToBatch(1).apply(image.toIterator)

    var t1 = miniBatch1.next()
    var t2 = miniBatch2.next()
    t1.getInput() should be (t2.getInput())
    t1.getTarget().asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget())
    t1.getInput().asInstanceOf[Tensor[Float]].size() should be (Array(1, 3, 32, 32))
    t1.getTarget().asInstanceOf[Tensor[Float]].valueAt(1) should be (image1.label())

    t1 = miniBatch1.next()
    t2 = miniBatch2.next()
    t1.getInput should be (t2.getInput)
    t1.getTarget.asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget)

    t1 = miniBatch1.next()
    t2 = miniBatch2.next()
    t1.getInput should be (t2.getInput)
    t1.getTarget.asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget)

    miniBatch1.hasNext should be (false)
    miniBatch2.hasNext should be (false)
  }

  "GreyImgToSample" should "be correct" in {
    RNG.setSeed(100)
    var data = new Array[Float](32 * 32)
    data = for (d <- data) yield RNG.uniform(1.0, 2.0).toFloat
    val image1 = new LabeledGreyImage(data, 32, 32, 1.0f)
    data = for (d <- data) yield RNG.uniform(1.1, 2.1).toFloat
    val image2 = new LabeledGreyImage(data, 32, 32, 2.0f)
    data = for (d <- data) yield RNG.uniform(1.2, 2.2).toFloat
    val image3 = new LabeledGreyImage(data, 32, 32, 3.0f)

    val image = Array(image1, image2, image3)
    val toSample = GreyImgToSample() -> SampleToMiniBatch(1)
    val miniBatch1 = toSample(image.toIterator)
    val miniBatch2 = GreyImgToBatch(1).apply(image.toIterator)

    var t1 = miniBatch1.next()
    var t2 = miniBatch2.next()
    t1.getInput should be (t2.getInput)
    t1.getTarget.asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget)
    t1.getInput.asInstanceOf[Tensor[Float]].size() should be (Array(1, 32, 32))
    t1.getTarget.asInstanceOf[Tensor[Float]].valueAt(1) should be (image1.label())
    t1.getInput.asInstanceOf[Tensor[Float]].storage().array() should be (image1.content)

    t1 = miniBatch1.next()
    t2 = miniBatch2.next()
    t1.getInput should be (t2.getInput)
    t1.getTarget.asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget)

    t1 = miniBatch1.next()
    t2 = miniBatch2.next()
    t1.getInput should be (t2.getInput)
    t1.getTarget.asInstanceOf[Tensor[Float]].squeeze() should be (t2.getTarget)

    miniBatch1.hasNext should be (false)
    miniBatch2.hasNext should be (false)
  }
}
