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

package com.intel.analytics.zoo.examples.localEstimator

import java.io.ByteArrayOutputStream
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.dataset.image.LabeledGreyImage
import com.intel.analytics.zoo.pipeline.estimator.EstimateSupportive
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.slf4j.LoggerFactory

object MnistDataLoader extends ImageProcessing with EstimateSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  def loadTrainData(imageDirPath: String): Array[LabeledGreyImage] = {
    val records = timing("load bytes from bin") {
      val trainDataPath = imageDirPath + "/train-images-idx3-ubyte"
      val trainLabelPath = imageDirPath + "/train-labels-idx1-ubyte"
      loadData(trainDataPath, trainLabelPath)
    }
    logger.info(s"${records.length} train data loaded")
    val labeledGreyImages = timing("transfer bytes to labeledGreyImages") {
      records.map(record => bytesToGreyImage(record, 28, 28))
    }
    val normalized = timing("normalize labeledGreyImages") {
      labeledGreyImages.map(record => greyImgNormalize(record, trainMean, trainStd))
    }
    normalized
  }

  def loadTestData(imageDirPath: String): Array[LabeledGreyImage] = {
    val records = timing("load bytes from bin") {
      val validationDataPath = imageDirPath + "/t10k-images-idx3-ubyte"
      val validationLabelPath = imageDirPath + "/t10k-labels-idx1-ubyte"
      MnistDataLoader.loadData(validationDataPath, validationLabelPath)
    }
    logger.info(s"${records.length} train data loaded")
    val labeledGreyImages = timing("transfer bytes to labeledGreyImages") {
      records.map(record => bytesToGreyImage(record, 28, 28))
    }
    val normalized = timing("normalize labeledGreyImages") {
      labeledGreyImages.map(record => greyImgNormalize(record, testMean, testStd))
    }
    normalized
  }

  def loadData(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val featureBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049, "labelMagicNumber should be 2049")
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051, "featureMagicNumber should be 2051")

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }

  private def readHdfsByte(fileName: String): Array[Byte] = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = FileSystem.newInstance(new URI(fileName), new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }
}


