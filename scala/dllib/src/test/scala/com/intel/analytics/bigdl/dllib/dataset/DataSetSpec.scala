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

import java.io.{ByteArrayInputStream, File, FileInputStream}
import java.nio.file.Paths
import java.util.concurrent.{Callable, Executors}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.segmentation.{COCODataset, COCOPoly, COCORLE, PolyMasks, RLEMasks}
import com.intel.analytics.bigdl.models.utils.COCOSeqFileGenerator
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, RoiImageInfo}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, SparkContextLifeCycle, TestUtils}
import java.awt.image.DataBufferByte
import javax.imageio.ImageIO
import org.apache.hadoop.io.Text
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class DataSetSpec extends SparkContextLifeCycle with Matchers {
  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "DataSetSpec"

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "COCODataset" should "correctly transform into sequence file" in {
    val resource = getClass().getClassLoader().getResource("coco")

    // load data from JSON for comparision
    val ds = COCODataset.load(processPath(resource.getPath())
      + File.separator + "cocomini.json")
    val index = ds.images.toIterator.map(im => (im.fileName, im)).toMap

    val dataSetFolder = processPath(resource.getPath()) + File.separator
    val tmpFile = java.io.File.createTempFile("UnitTest", System.nanoTime().toString)
    require(tmpFile.delete())
    require(tmpFile.mkdir())
    COCOSeqFileGenerator.main(Array("-f", dataSetFolder, "-o", tmpFile.getPath, "-p", "4",
      "-b", "2", "-m", dataSetFolder + "cocomini.json"))

    // write done, now read and check
    DataSet.SeqFileFolder.filesToRoiImageFeatures(tmpFile.getPath, sc).toDistributed()
      .data(false)
      .map(imf => {
        (imf(ImageFeature.uri).asInstanceOf[String], imf.getOriginalSize, imf.getLabel[RoiLabel],
          imf[Tensor[Float]](RoiImageInfo.ISCROWD), imf[Array[Byte]](ImageFeature.bytes))
      })
      .collect()
      .foreach({ case (uri, size, label, iscrowd, bytes) =>
        val img = index(uri)
        require(size == (img.height, img.width, 3))
        require(label.masks.length == img.annotations.length)
        require(java.util.Arrays.equals(iscrowd.toArray(),
          img.annotations.map(a => if (a.isCrowd) 1f else 0f).toArray))
        img.annotations.zipWithIndex.foreach { case (ann, idx) =>
          label.masks(idx) match {
            case rle: RLEMasks =>
              val realArr = ann.segmentation.asInstanceOf[COCORLE].counts
              val seqArr = rle.counts
              require(java.util.Arrays.equals(realArr, seqArr))
            case poly: PolyMasks =>
              val realArr = ann.segmentation.asInstanceOf[PolyMasks].poly.flatten
              val seqArr = poly.poly.flatten
              require(java.util.Arrays.equals(realArr, seqArr))
          }

          val bb = label.bboxes.narrow(1, idx + 1, 1).squeeze().toArray()
          val annbb = Array(ann.bbox._1, ann.bbox._2,
            ann.bbox._3, ann.bbox._4)
          require(java.util.Arrays.equals(bb, annbb))
        }

        // label checking done, now check the image data
        val inputStream = new FileInputStream(dataSetFolder + uri)
        val image = ImageIO.read(inputStream)
        val rawdata = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
        require(java.util.Arrays.equals(rawdata, bytes))
      })
  }


  "mnist data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSet = DataSet.array(com.intel.analytics.bigdl.models.lenet.Utils.load(
      processPath(resource.getPath()) + File.separator + "t10k-images.idx3-ubyte",
      processPath(resource.getPath()) + File.separator + "t10k-labels.idx1-ubyte")
    )
    dataSet.size() should be(10000)
    var iter = dataSet.data(train = false)
    iter.map(_.label).min should be(1.0f)
    iter = dataSet.data(train = false)
    iter.map(_.label).max should be(10.0f)
  }

  "mnist rdd data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSet = DataSet.array(com.intel.analytics.bigdl.models.lenet.Utils.load(
      processPath(resource.getPath()) + File.separator + "t10k-images.idx3-ubyte",
      processPath(resource.getPath()) + File.separator + "t10k-labels.idx1-ubyte"
    ), sc)

    dataSet.size() should be(10000)
    var rdd = dataSet.data(train = false)
    rdd.map(_.label).min should be(1.0f)
    rdd = dataSet.data(train = false)
    rdd.map(_.label).max should be(10.0f)
  }

  "cifar data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("cifar")
    val dataSet = DataSet.ImageFolder.images(Paths.get(processPath(resource.getPath())),
      BGRImage.NO_SCALE)
    dataSet.size() should be(7)
    val labelMap = LocalImageFiles.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("airplane") should be(1)
    labelMap("deer") should be(2)

    val iter = dataSet.toLocal().data(train = false)
    val img1 = iter.next()
    img1.label() should be(1f)
    img1.content(2) should be(234 / 255f)
    img1.content(1) should be(125 / 255f)
    img1.content(0) should be(59 / 255f)
    img1.content((22 + 4 * 32) * 3 + 2) should be(253 / 255f)
    img1.content((22 + 4 * 32) * 3 + 1) should be(148 / 255f)
    img1.content((22 + 4 * 32) * 3) should be(31 / 255f)
    val img2 = iter.next()
    img2.label() should be(1f)
    val img3 = iter.next()
    img3.label() should be(2f)
    val img4 = iter.next()
    img4.label() should be(2f)
    img4.content((9 + 8 * 32) * 3 + 2) should be(40 / 255f)
    img4.content((9 + 8 * 32) * 3 + 1) should be(51 / 255f)
    img4.content((9 + 8 * 32) * 3) should be(37 / 255f)
    val img5 = iter.next()
    img5.label() should be(2f)
    val img6 = iter.next()
    img6.label() should be(2f)
    val img7 = iter.next()
    img7.label() should be(1f)
  }

  "cifar rdd data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("cifar")
    val dataSet = DataSet.ImageFolder.images(Paths.get(processPath(resource.getPath())),
      sc, BGRImage.NO_SCALE)
    dataSet.size() should be(7)
    val labelMap = LocalImageFiles.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("airplane") should be(1)
    labelMap("deer") should be(2)

    val rdd = dataSet.toDistributed().data(train = false)
    rdd.filter(_.label() == 1f).count() should be(3)
    rdd.filter(_.label() == 2f).count() should be(4)
    val images = rdd.map(_.clone())
      .filter(_.label() == 1f)
      .collect()
      .sortWith(_.content(0) < _.content(0))
    val img1 = images(1)
    img1.label() should be(1f)
    img1.content(2) should be(234 / 255f)
    img1.content(1) should be(125 / 255f)
    img1.content(0) should be(59 / 255f)
    img1.content((22 + 4 * 32) * 3 + 2) should be(253 / 255f)
    img1.content((22 + 4 * 32) * 3 + 1) should be(148 / 255f)
    img1.content((22 + 4 * 32) * 3) should be(31 / 255f)

    val images2 = rdd.map(_.clone())
      .filter(_.label() == 2)
      .collect()
      .sortWith(_.content(0) < _.content(0))
    val img4 = images2(0)
    img4.label() should be(2f)
    img4.content((9 + 8 * 32) * 3 + 2) should be(40 / 255f)
    img4.content((9 + 8 * 32) * 3 + 1) should be(51 / 255f)
    img4.content((9 + 8 * 32) * 3) should be(37 / 255f)
  }

  "imagenet data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val dataSet = DataSet.ImageFolder.paths(Paths.get(processPath(resource.getPath())))
    dataSet.size() should be(11)

    val labelMap = LocalImageFiles.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("n02110063") should be(1)
    labelMap("n04370456") should be(2)
    labelMap("n15075141") should be(3)
    labelMap("n99999999") should be(4)

    val pathToImage = LocalImgReader(BGRImage.NO_SCALE)
    val imageDataSet = dataSet -> pathToImage

    val images = imageDataSet.toLocal().data(train = false)
      .map(_.clone())
      .toArray
      .sortWith(_.content(0) < _.content(0))
    val labels = images.map(_.label())
    labels.mkString(",") should be("2.0,3.0,1.0,4.0,1.0,1.0,4.0,3.0,4.0,3.0,2.0")

    images(6).content((100 + 100 * 213) * 3 + 2) should be(35 / 255f)
    images(6).content((100 + 100 * 213) * 3 + 1) should be(30 / 255f)
    images(6).content((100 + 100 * 213) * 3) should be(36 / 255f)
    val path1 = java.io.File.createTempFile("UnitTest", "datasource1.jpg").getAbsolutePath
    images(6).save(path1)
    println(s"save test image to $path1")

    images(8).content((100 + 100 * 556) * 3 + 2) should be(24 / 255f)
    images(8).content((100 + 100 * 556) * 3 + 1) should be(24 / 255f)
    images(8).content((100 + 100 * 556) * 3) should be(24 / 255f)
    val path2 = java.io.File.createTempFile("UnitTest", "datasource2.jpg").getAbsolutePath
    images(8).save(path2)
    println(s"save test image to $path2")
  }

  "imagenet sequence data source" should "load image correct" in {
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

    files.length should be(6)

    val imageIter = (DataSet.SeqFileFolder.paths(Paths.get(tmpFile.getAbsolutePath()), 11)
      -> LocalSeqFileToBytes() -> BytesToBGRImg()).toLocal().data(train = false)

    val img = imageIter.next()
    img.label() should be(4f)
    img.content((100 + 100 * 213) * 3 + 2) should be(35 / 255f)
    img.content((100 + 100 * 213) * 3 + 1) should be(30 / 255f)
    img.content((100 + 100 * 213) * 3) should be(36 / 255f)
    imageIter.next()
    img.label() should be(4f)
    img.content((100 + 100 * 556) * 3 + 2) should be(24 / 255f)
    img.content((100 + 100 * 556) * 3 + 1) should be(24 / 255f)
    img.content((100 + 100 * 556) * 3) should be(24 / 255f)
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.next()
    imageIter.hasNext should be(false)
  }

  "ImageNet data source" should "load image correct with parallel process" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val labelMap = LocalImageFiles.readLabels(Paths.get(processPath(resource.getPath())))
    val dataSet = DataSet.ImageFolder.images(Paths.get(processPath(resource.getPath())),
      BGRImage.NO_SCALE)

    val iter = dataSet.toLocal().data(train = false)
    val parallel = 10
    val syncPool = Executors.newFixedThreadPool(parallel)
    val tasks = (0 until parallel).map(pid => {
      syncPool.submit(new Callable[Int] {
        override def call(): Int = {
          var cc = 0
          while (iter.hasNext) {
            val img = iter.next()
            if (null != img) {
              cc += img.label().toInt
            }
            Thread.sleep(1)
          }
          cc
        }
      })
    })
    val count = tasks.map(_.get()).reduce(_ + _)
    count should be (28)
    syncPool.shutdown()
  }

  "image preprocess" should "be same with torch result" in {
    val originMode = Engine.localMode
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
    val resourceImageNet = getClass().getClassLoader().getResource("imagenet")
    def test(imgFolder: String, imgFileName: String, tensorFile: String): Unit = {
      val img1Path = Paths.get(processPath(resourceImageNet.getPath()), imgFolder, imgFileName)
      val iter = (DataSet.array(Array(LocalLabeledImagePath(1.0f, img1Path)))
        -> LocalImgReader()
        -> BGRImgCropper(224, 224)
        -> HFlip()
        -> BGRImgNormalizer((0.4, 0.5, 0.6), (0.1, 0.2, 0.3))
        -> BGRImgToBatch(1)
        ).toLocal().data(train = false)
      val image1 = iter.next().getInput.toTensor[Float]

      val resourceTorch = getClass().getClassLoader().getResource("torch")
      val tensor1Path = Paths.get(processPath(resourceTorch.getPath()), tensorFile)
      val tensor1 = com.intel.analytics.bigdl.utils.File.loadTorch[Tensor[Float]](
        tensor1Path.toString).addSingletonDimension()
      image1.size() should be(tensor1.size())
      image1.map(tensor1, (a, b) => {
        a should be(b +- 0.0001f)
        b
      })
    }
    RandomGenerator.RNG.setSeed(100)
    test("n02110063", "n02110063_11239.JPEG", "n02110063_11239.t7")
    RandomGenerator.RNG.setSeed(100)
    test("n04370456", "n04370456_5753.JPEG", "n04370456_5753.t7")
    RandomGenerator.RNG.setSeed(100)
    test("n15075141", "n15075141_38508.JPEG", "n15075141_38508.t7")
    RandomGenerator.RNG.setSeed(100)
    test("n99999999", "n03000134_4970.JPEG", "n03000134_4970.t7")
    System.clearProperty("bigdl.localMode")
  }

  "RDD from DataSet" should "give different position every time" in {
    val data = (1 to 4).toArray
    val trainRDD = DataSet.rdd(sc.parallelize(data, 1).mapPartitions(_ => {
      RandomGenerator.RNG.setSeed(100)
      (1 to 100).iterator
    })).data(train = true)
    trainRDD.mapPartitions(iter => {
      Iterator.single(iter.next())
    }).collect()(0) should be(55)

    trainRDD.mapPartitions(iter => {
      Iterator.single(iter.next())
    }).collect()(0) should be(68)

    trainRDD.mapPartitions(iter => {
      Iterator.single(iter.next())
    }).collect()(0) should be(28)

  }

  "RDD DataSet" should "be good for sorted buffer with two partition" in {
    val count = 100
    val data = new Array[Sample[Float]](count)
    var i = 1
    while (i <= count) {
      val input = Tensor[Float](3, 28, 28).apply1(e => Random.nextFloat())
      val target = Tensor[Float](1).fill(i.toFloat)
      data(i-1) = Sample(input, target)
      i += 1
    }

    val partitionNum = 2
    val batchSize = 5
    val groupSize = 5
    val dataSet1 = new CachedDistriDataSet[Sample[Float]](
      sc.parallelize(data, partitionNum)
        .coalesce(partitionNum, true)
        .mapPartitions(iter => {
          val tmp = iter.toArray
          Iterator.single(tmp)
        }).setName("cached dataset")
        .cache(),
      true,
      groupSize
    )

    val dataSet = dataSet1.transform(SampleToMiniBatch(batchSize))
    val rdd = dataSet.toDistributed().data(train = true)
    rdd.partitions.size should be (partitionNum)
    val rddData = rdd.mapPartitions(iter => {
      Iterator.single(iter.next().getTarget.toTensor[Float])
    })

    i = 0
    while (i < 100) {
      val label = rddData.collect()(0).storage().array()
      label.reduce((l, f) => if (l < f) f else 10000) should not be (10000)
      i += 1
    }
  }

  "RDD test DataSet" should "be same to the original data with one partition" in {
    val count = 100
    val data = new Array[Sample[Float]](count)
    var i = 1
    while (i <= count) {
      val input = Tensor[Float](3, 28, 28).apply1(e => Random.nextFloat())
      val target = Tensor[Float](1).fill(i.toFloat)
      data(i-1) = Sample(input, target)
      i += 1
    }
    val partitionNum = 1
    val dataRDD = sc.parallelize(data, partitionNum).coalesce(partitionNum, true)
    val dataSet = DataSet.sortRDD(dataRDD, true, 10)
    val rdd = dataSet.toDistributed().data(train = false)
    val localData = rdd.collect()

    i = 0
    while (i < localData.length) {
      localData(i) should be (data(i))
      i += 1
    }
  }

  "transformRDD" should "be correct" in {
    TestUtils.cancelOnWindows()
    val resource = getClass().getClassLoader().getResource("imagenet")
    val tmpFile = java.io.File.createTempFile("UnitTest", System.nanoTime().toString)
    require(tmpFile.delete())
    require(tmpFile.mkdir())

    // Convert the test imagenet files to seq files
    val files = (DataSet.ImageFolder.paths(Paths.get(processPath(resource.getPath())))
      -> LocalImgReaderWithName(BGRImage.NO_SCALE)
      -> BGRImgToLocalSeqFile(100, Paths.get(tmpFile.getAbsolutePath(), "imagenet"))
      ).toLocal().data(train = false).map(s => {
      println(s);
      s
    }).toArray

    val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
    val rddData = sc.sequenceFile(tmpFile.getAbsolutePath(), classOf[Text],
      classOf[Text], partitionNum).map(image => {
        ByteRecord(image._2.copyBytes(), DataSet.SeqFileFolder.readLabel(image._1).toFloat)
      })
    val transformer = BytesToBGRImg()
    val imageIter = transformer(rddData)

    val result = imageIter.map(_.clone()).collect().sortBy(_.label())
    result.length should be(11)
    var img = result(0)
    img.label() should be(1f)
    img.content((100 + 100 * 213) * 3 + 2) should be(17 / 255f)
    img.content((100 + 100 * 213) * 3 + 1) should be(27 / 255f)
    img.content((100 + 100 * 213) * 3) should be(26 / 255f)

    img = result(8)
    img.label() should be(4f)
    img.content((100 + 100 * 213) * 3 + 2) should be(35 / 255f)
    img.content((100 + 100 * 213) * 3 + 1) should be(30 / 255f)
    img.content((100 + 100 * 213) * 3) should be(36 / 255f)
  }
}
