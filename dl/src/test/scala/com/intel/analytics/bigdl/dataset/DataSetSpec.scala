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

import java.io.File
import java.nio.file.Paths

import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FlatSpec, Matchers}

class DataSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc: SparkContext = null

  before {
    sc = new SparkContext("local[1]", "DataSetSpec")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "mnist data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSource = MNIST.LocalDataSet(
      Paths.get(processPath(resource.getPath()) + File.separator, "t10k-images.idx3-ubyte"),
      Paths.get(processPath(resource.getPath()) + File.separator, "t10k-labels.idx1-ubyte"),
      looped = false
    )
    dataSource.size() should be(10000)
    val iter = dataSource.data()
    iter.map(_._1).min should be(1.0f)
    dataSource.reset()
    iter.map(_._1).max should be(10.0f)
  }

  "mnist rdd data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSource = MNIST.RDDDataSet(
      Paths.get(processPath(resource.getPath()) + File.separator, "t10k-images.idx3-ubyte"),
      Paths.get(processPath(resource.getPath()) + File.separator, "t10k-labels.idx1-ubyte"),
      looped = false,
      sc,
      4
    )
    dataSource.size() should be(10000)
    val rdd = dataSource.data()
    rdd.map(_._1).min should be(1.0f)
    dataSource.reset()
    rdd.map(_._1).max should be(10.0f)
  }

  "cifar data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("cifar")
    val dataSource = Cifar.LocalDataSet(Paths.get(processPath(resource.getPath())),
      looped = false)
    val imgDataSource = (dataSource -> ArrayByteToRGBImage(255.0f))
    dataSource.size() should be(7)
    val labelMap = DirectoryAsLabel.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("airplane") should be(1)
    labelMap("deer") should be(2)

    val iter = imgDataSource.data()
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
    val dataSource = Cifar.RDDDataSet(Paths.get(processPath(resource.getPath())),
      false, sc, 4)
    val imgDataSource = (dataSource -> ArrayByteToRGBImage(255.0f))
    dataSource.size() should be(7)
    val labelMap = DirectoryAsLabel.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("airplane") should be(1)
    labelMap("deer") should be(2)

    imgDataSource.size() should be(7)
    val rdd = imgDataSource.data()
    rdd.filter(_.label() == 1f).count() should be(3)
    rdd.filter(_.label() == 2f).count() should be(4)
    val images = rdd.filter(_.label() == 1f).sortBy(_.content(0)).collect()
    val img1 = images(1)
    img1.label() should be(1f)
    img1.content(2) should be(234 / 255f)
    img1.content(1) should be(125 / 255f)
    img1.content(0) should be(59 / 255f)
    img1.content((22 + 4 * 32) * 3 + 2) should be(253 / 255f)
    img1.content((22 + 4 * 32) * 3 + 1) should be(148 / 255f)
    img1.content((22 + 4 * 32) * 3) should be(31 / 255f)

    val images2 = rdd.filter(_.label() == 2).sortBy(_.content(0)).collect()
    val img4 = images2(0)
    img4.label() should be(2f)
    img4.content((9 + 8 * 32) * 3 + 2) should be(40 / 255f)
    img4.content((9 + 8 * 32) * 3 + 1) should be(51 / 255f)
    img4.content((9 + 8 * 32) * 3) should be(37 / 255f)
  }

  "imagenet data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val dataSource = ImageNet.PathDataSet(Paths.get(processPath(resource.getPath())), looped =
      false)
    dataSource.size() should be(11)

    val labelMap = DirectoryAsLabel.readLabels(Paths.get(processPath(resource.getPath())))
    labelMap("n02110063") should be(1)
    labelMap("n04370456") should be(2)
    labelMap("n15075141") should be(3)
    labelMap("n99999999") should be(4)

    var pathToImage = PathToRGBImage(RGBImage.NO_SCALE)
    var imageDataSource = dataSource -> pathToImage

    var iter = imageDataSource.data()
    val img1 = iter.next()
    img1.label() should be(4f)
    img1.content((100 + 100 * 213) * 3 + 2) should be(35 / 255f)
    img1.content((100 + 100 * 213) * 3 + 1) should be(30 / 255f)
    img1.content((100 + 100 * 213) * 3) should be(36 / 255f)
    val path1 = java.io.File.createTempFile("UnitTest", "datasource1.jpg").getAbsolutePath
    img1.save(path1)
    println(s"save test image to $path1")

    val img2 = iter.next()
    img2.label() should be(4f)
    img2.content((100 + 100 * 556) * 3 + 2) should be(24 / 255f)
    img2.content((100 + 100 * 556) * 3 + 1) should be(24 / 255f)
    img2.content((100 + 100 * 556) * 3) should be(24 / 255f)
    val path2 = java.io.File.createTempFile("UnitTest", "datasource2.jpg").getAbsolutePath
    img1.save(path2)
    println(s"save test image to $path2")

    pathToImage = PathToRGBImage(256)
    imageDataSource = dataSource -> pathToImage
    iter = imageDataSource.data()

    val img3 = iter.next()
    img3.label() should be(1f)
    (img3.width() == 256 || img3.height() == 256) should be(true)
    val path3 = java.io.File.createTempFile("UnitTest", "datasource3.jpg").getAbsolutePath
    img3.save(path3)
    println(s"save test image to $path3")

    val img4 = iter.next()
    img4.label() should be(1f)
    (img4.width() == 256 || img4.height() == 256) should be(true)

    val img5 = iter.next()
    img5.label() should be(1f)
    (img5.width() == 256 || img5.height() == 256) should be(true)

    val img6 = iter.next()
    img6.label() should be(4f)
    (img6.width() == 256 || img6.height() == 256) should be(true)

    val img7 = iter.next()
    img7.label() should be(2f)
    (img7.width() == 256 || img7.height() == 256) should be(true)

    val img8 = iter.next()
    img8.label() should be(2f)
    (img8.width() == 256 || img8.height() == 256) should be(true)

    val img9 = iter.next()
    img9.label() should be(3f)
    (img9.width() == 256 || img9.height() == 256) should be(true)

    val img10 = iter.next()
    img10.label() should be(3f)
    (img10.width() == 256 || img10.height() == 256) should be(true)

    val img11 = iter.next()
    img11.label() should be(3f)
    (img11.width() == 256 || img11.height() == 256) should be(true)
  }

  "imagenet seq data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val tmpFile = java.io.File.createTempFile("UnitTest", System.nanoTime().toString)
    require(tmpFile.delete())
    require(tmpFile.mkdir())

    // Convert the test imagenet files to seq files
    val files = (ImageNet.PathDataSet(Paths.get(processPath(resource.getPath())), looped = false)
      -> PathToRGBImage(RGBImage.NO_SCALE)
      -> RGBImageToSequentialFile(2, Paths.get(tmpFile.getAbsolutePath(), "imagenet"))
      ).data().map(s => {
      println(s); s
    }).toArray

    files.length should be(6)

    val imageIter = (ImageNet.LocalSeqDataSet(Paths.get(tmpFile.getAbsolutePath()), 11, false)
      -> SeqFileToArrayByte() -> ArrayByteToRGBImage()).data()

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

  "imagenet rdd seq data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val tmpFile = java.io.File.createTempFile("UnitTest", System.nanoTime().toString)
    require(tmpFile.delete())
    require(tmpFile.mkdir())

    // Convert the test imagenet files to seq files
    val files = (ImageNet.PathDataSet(Paths.get(processPath(resource.getPath())), looped = false)
      -> PathToRGBImage(RGBImage.NO_SCALE)
      -> RGBImageToSequentialFile(2, Paths.get(tmpFile.getAbsolutePath(), "imagenet"))
      ).data().map(s => {
      println(s); s
    }).toArray

    files.length should be(6)

    val imageDS = (ImageNet.RDDSeqDataSet(tmpFile.getAbsolutePath().toString, sc, 1000, false, 4)
      -> ArrayByteToRGBImage())
    imageDS.size() should be(11)
    val images = imageDS.data().filter(_.label() == 4f).map(_.clone()).collect()
      .sortWith(_.content(0) < _.content(0))
    images.length should be(3)

    val img1 = images(1)
    img1.label() should be(4f)
    img1.content((100 + 100 * 213) * 3 + 2) should be(35 / 255f)
    img1.content((100 + 100 * 213) * 3 + 1) should be(30 / 255f)
    img1.content((100 + 100 * 213) * 3) should be(36 / 255f)
    val img2 = images(2)
    img2.label() should be(4f)
    img2.content((100 + 100 * 556) * 3 + 2) should be(24 / 255f)
    img2.content((100 + 100 * 556) * 3 + 1) should be(24 / 255f)
    img2.content((100 + 100 * 556) * 3) should be(24 / 255f)
  }
}
