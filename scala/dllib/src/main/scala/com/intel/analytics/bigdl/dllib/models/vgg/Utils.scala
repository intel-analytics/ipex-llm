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

package com.intel.analytics.bigdl.models.vgg

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.image.{CropCenter, CropRandom}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, MiniBatch}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.transform.vision.image.{MTImageFeatureToBatch, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object Utils {
  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  case class TrainParams(
    folder: String = "./",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    summaryPath: Option[String] = None,
    batchSize: Int = 112,
    maxEpoch: Int = 90,
    overWriteCheckpoint: Boolean = false,
    learningRate: Double = 0.01,
    weightDecay: Double = 0.0005,
    graphModel: Boolean = false,
    maxIteration: Int = 40000,
    momentum: Double = 0.9,
    dampening: Double = 0.0,
    nesterov: Boolean = true,
    classNumber: Int = 1000,
    env: String = "local",
    checkpointIteration: Int = 1000,
    maxLr: Double = 0.06,
    warmupEpoch: Option[Int] = None,
    gradientL2NormThreshold: Option[Double] = None,
    optimizerVersion: Option[String] = None
  )

  val trainParser = new OptionParser[TrainParams]("BigDL Vgg Example") {
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model and state")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[String]("summary")
      .text("where to store the training summary")
      .action((x, c) => c.copy(summaryPath = Some(x)))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action( (_, c) => c.copy(overWriteCheckpoint = true) )
    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Unit]('g', "graphModel")
      .text("use graph model")
      .action((x, c) => c.copy(graphModel = true))
    opt[Int]('i', "maxIteration")
      .text("iteration numbers")
      .action((x, c) => c.copy(maxIteration = x))
    opt[Int]("classNum")
      .text("class number")
      .action((x, c) => c.copy(classNumber = x))
    opt[Int]("checkpointIteration")
      .text("checkpoint interval of iterations")
      .action((x, c) => c.copy(checkpointIteration = x))
    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("maxLr")
      .text("max Lr after warm up")
      .action((x, c) => c.copy(maxLr = x))
    opt[Int]("warmupEpoch")
      .text("warm up epoch numbers")
      .action((x, c) => c.copy(warmupEpoch = Some(x)))
    opt[Double]("gradientL2NormThreshold")
      .text("gradient L2-Norm threshold")
      .action((x, c) => c.copy(gradientL2NormThreshold = Some(x)))
    opt[String]("optimizerVersion")
      .text("state optimizer version")
      .action((x, c) => c.copy(optimizerVersion = Some(x)))
  }

  case class TestParams(
    folder: String = "./",
    model: String = "",
    batchSize: Int = 112
  )

  val testParser = new OptionParser[TestParams]("BigDL Vgg on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  private[bigdl] def loadTrain(dataFile: String): Array[ByteRecord] = {
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

  private[bigdl] def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = dataFile + "/test_batch.bin"
    load(testFile, result)
    result.toArray
  }

  /**
   * load cifar data.
   * read cifar from hdfs if data folder starts with "hdfs:", otherwise form local file.
   * @param featureFile
   * @param result
   */
  private[bigdl] def load(featureFile: String, result : ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = if (featureFile.startsWith(File.hdfsPrefix)) {
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

  private type BatchDataSet = DataSet[MiniBatch[Float]]
  def valDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): BatchDataSet = {
    DataSet.SeqFileFolder.filesToImageFeatureDataset(path, sc, 1000).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, false, CropCenter) ->
          ChannelScaledNormalizer(104, 117, 124, 1) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): BatchDataSet = {
    DataSet.SeqFileFolder.filesToImageFeatureDataset(path, sc, 1000).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelScaledNormalizer(104, 117, 124, 1) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }
}

