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

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.zoo.pipeline.estimator.EstimateSupportive
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

object Cifar10DataLoader extends ImageProcessing with EstimateSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  val resizeW = 32
  val resizeH = 32

  def loadTrainData(imageDirPath: String): Array[LabeledBGRImage] = {
    val records = timing("load bytes from bin") {
      loadTrain(imageDirPath)
    }
    logger.info(s"${records.length} train data loaded")
    val labeledBGRImages = timing("transfer bytes to bgrImages") {
      records.map(record => bytesToLabeledBGRImage(record, resizeW, resizeH))
    }

    val normalized = timing("normalize the images") {
      labeledBGRImages.map(image => bgrImgNormalize(image, trainMean, trainStd))
    }
    val hFliped = timing("hflip the images") {
      normalized.map(image => hFlip(image, 0.5))
    }
    val randomCropped = timing("randomCrop the images") {
      hFliped.map(image => bgrImageRandomCrop(image, 32, 32, 4))
    }
    randomCropped
  }

  def loadTestData(imageDirPath: String): Array[LabeledBGRImage] = {
    val records = timing("load bytes from bin") {
      loadTest(imageDirPath)
    }
    logger.info(s"${records.length} test data loaded")
    val labeledBGRImages = timing("transfer bytes to bgrImages") {
      records.map(record => bytesToLabeledBGRImage(record, resizeW, resizeH))
    }
    val normalized = timing("normalize the images") {
      labeledBGRImages.map(image => bgrImgNormalize(image, testMean, testStd))
    }
    normalized
  }

  def loadTrain(dataFile: String): Array[ByteRecord] = {
    val allFiles = Array(
      dataFile + "/data_batch_1.bin",
      dataFile + "/data_batch_2.bin",
      dataFile + "/data_batch_3.bin",
      dataFile + "/data_batch_4.bin",
      dataFile + "/data_batch_5.bin"
    )
    val result = new ArrayBuffer[ByteRecord]()
    allFiles.foreach(load(_, result))
    result.toArray
  }

  def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = dataFile + "/test_batch.bin"
    load(testFile, result)
    result.toArray
  }

  private def load(featureFile: String, result: ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }

    val featureArray = featureBuffer.array()
    val featureCount = featureArray.length / (rowNum * colNum * 3 + 1)

    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum * 3 + bufferOffset))
      val byteBuffer = ByteBuffer.wrap(img)
      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i * imageOffset).toFloat
      var y = 0
      val start = i * imageOffset + 1
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img((x + y * colNum) * 3 + 2 + bufferOffset) =
            featureArray(start + x + y * colNum)
          img((x + y * colNum) * 3 + 1 + bufferOffset) =
            featureArray(start + x + y * colNum + channelOffset)
          img((x + y * colNum) * 3 + bufferOffset) =
            featureArray(start + x + y * colNum + 2 * channelOffset)
          x += 1
        }
        y += 1
      }
      result.append(ByteRecord(img, label + 1.0f))
      i += 1
    }
  }
}
