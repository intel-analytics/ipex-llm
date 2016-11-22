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

package com.intel.analytics.sparkdl.dataset

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.Executors

import com.intel.analytics.sparkdl.models.imagenet.{AlexNet, GoogleNet_v1}
import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Criterion, Module}
import com.intel.analytics.sparkdl.optim.SGD.LearningRateSchedule
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.T
import scopt.OptionParser

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}

object ImageNetSeqFileGenerator {

  case class ImageNetSeqFileGeneratorParams(
    folder: String = ".",
    output: String = ".",
    parallel: Int = 1,
    blockSize: Int = 12800,
    train: Boolean = true,
    validate: Boolean = true
  )

  private val parser = new OptionParser[ImageNetSeqFileGeneratorParams]("Spark-DL ImageNet " +
    "Sequence File Generator") {
    head("Spark-DL ImageNet Sequence File Generator")
    opt[String]('f', "folder")
      .text("where you put the ImageNet data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('o', "output folder")
      .text("where you put the generated seq files")
      .action((x, c) => c.copy(output = x))
    opt[Int]('p', "parallel")
      .text("parallel num")
      .action((x, c) => c.copy(parallel = x))
    opt[Int]('b', "blockSize")
      .text("block size")
      .action((x, c) => c.copy(blockSize = x))
    opt[Unit]('t', "trainOnly")
      .text("only generate train data")
      .action((_, c) => c.copy(validate = false))
    opt[Unit]('v', "validationOnly")
      .text("only generate validation data")
      .action((_, c) => c.copy(train = false))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new ImageNetSeqFileGeneratorParams()).map(param => {
      if (param.train) {
        // Process train data
        println("Process train data...")
        val trainFolderPath = Paths.get(param.folder, "train")
        require(Files.isDirectory(trainFolderPath),
          s"${trainFolderPath} is not valid")
        val trainDataSource = new ImageNetDataSource(trainFolderPath, false)
        trainDataSource.shuffle()
        (0 until param.parallel).map(tid => {
          val workingThread = new Thread(new Runnable {
            override def run(): Unit = {
              val pipeline = trainDataSource -> PathToRGBImage(256) ->
                RGBImageToSequentialFile(param.blockSize, Paths.get(param.output, "train",
                  s"imagenet-seq-$tid"))
              while (pipeline.hasNext) {
                println(s"Generated file ${pipeline.next()}")
              }
            }
          })
          workingThread.setDaemon(false)
          workingThread.start()
          workingThread
        }).foreach(_.join())
      }

      if (param.validate) {
        // Process validation data
        println("Process validation data...")
        val validationFolderPath = Paths.get(param.folder, "val")
        require(Files.isDirectory(validationFolderPath),
          s"${validationFolderPath} is not valid")

        val validationDataSource = new ImageNetDataSource(validationFolderPath, false)
        validationDataSource.shuffle()
        (0 until param.parallel).map(tid => {
          val workingThread = new Thread(new Runnable {
            override def run(): Unit = {
              val pipeline = validationDataSource -> PathToRGBImage(256) ->
                RGBImageToSequentialFile(param.blockSize, Paths.get(param.output, "val",
                  s"imagenet-seq-$tid"))
              while (pipeline.hasNext) {
                println(s"Generated file ${pipeline.next()}")
              }
            }
          })
          workingThread.setDaemon(false)
          workingThread.start()
          workingThread
        }).foreach(_.join())
      }
    })

    println("Done")
  }
}

object ImageNetLocal {

  case class ImageNetLocalParam(
    folder: String = "./",
    net: String = "alexnet",
    cache: String = "./",
    parallel: Int = 1
  )

  case class Config(
    model: Module[Tensor[Float], Tensor[Float], Float],
    criterion: Criterion[Tensor[Float], Float],
    optimMethod: OptimMethod[Float],
    imageSize: Int,
    batchSize: Int,
    momentum: Double,
    weightDecay: Double,
    testTrigger: Trigger,
    cacheTrigger: Trigger,
    endWhen: Trigger,
    learningRate: Double,
    learningRateSchedule: LearningRateSchedule
  )

  private val configs = Map(
    "alexnet" -> Config(
      AlexNet[Float](classNum = 1000),
      new ClassNLLCriterion[Float](),
      new SGD[Float](),
      imageSize = 227,
      batchSize = 256,
      momentum = 0.9,
      weightDecay = 0.0005,
      testTrigger = Trigger.severalIteration(1000),
      cacheTrigger = Trigger.severalIteration(10000),
      endWhen = Trigger.maxIteration(450000),
      learningRate = 0.01,
      learningRateSchedule = SGD.Step(100000, 0.1)),
    "googlenetv1" -> Config(
      GoogleNet_v1[Float](classNum = 1000),
      new ClassNLLCriterion[Float](),
      new SGD[Float](),
      imageSize = 224,
      batchSize = 32,
      momentum = 0.9,
      weightDecay = 0.0002,
      testTrigger = Trigger.severalIteration(4000),
      cacheTrigger = Trigger.severalIteration(40000),
      endWhen = Trigger.maxIteration(2400000),
      learningRate = 0.01,
      learningRateSchedule = SGD.Poly(0.5, 2400000))
  )

  private val parser = new OptionParser[ImageNetLocalParam]("Spark-DL ImageNet Local Example") {
    head("Spark-DL ImageNet Local Example")
    opt[String]('f', "folder")
      .text("where you put the ImageNet data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('c', "cache")
      .text("where you put the model and state snapshot")
      .action((x, c) => c.copy(cache = x))
    opt[Int]('p', "parallel")
      .text("parallel num")
      .action((x, c) => c.copy(parallel = x))
    opt[String]('n', "net")
      .text("net type : alexnet | googlenetv1")
      .action((x, c) => c.copy(net = x.toLowerCase))
      .validate(v =>
        if (Set("alexnet", "googlenetv1").contains(v.toLowerCase())) {
          success
        } else {
          failure("Net type can only be alexnet | googlenetv1 in this example")
        }
      )
  }

  def main(args: Array[String]) {
    parser.parse(args, new ImageNetLocalParam()).map(param => {
      val config = configs(param.net)
      val trainDataSource = new ImageNetSeqDataSource(Paths.get(param.folder, "train"), 1281167,
        looped = true)
      val validationDataSource = new ImageNetSeqDataSource(Paths.get(param.folder, "val"),
        50000, looped = false)
      val fileTransformer = new SeqFileToArrayByte()
      val arrayToImage = ArrayByteToRGBImage()
      val cropper = RGBImageCropper(cropWidth = config.imageSize, cropHeight = config.imageSize)
      val normalizer = RGBImageNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
      val flipper = HFlip(0.5)
      val trainMultiThreadToTensor = MultiThreadRGBImageToSingleTensor[Path](
        width = configs(param.net).imageSize,
        height = configs(param.net).imageSize,
        threadNum = param.parallel,
        batchSize = config.batchSize,
        transformer = fileTransformer + arrayToImage + cropper + flipper + normalizer
      )

      val validationMultiThreadToTensor = MultiThreadRGBImageToSingleTensor[Path](
        width = configs(param.net).imageSize,
        height = configs(param.net).imageSize,
        threadNum = param.parallel,
        batchSize = config.batchSize,
        transformer = fileTransformer + arrayToImage + cropper + normalizer
      )

      val optimizer = new LocalOptimizer[Float](
        data = trainDataSource -> trainMultiThreadToTensor,
        validationData = validationDataSource -> validationMultiThreadToTensor,
        model = config.model,
        criterion = config.criterion,
        optimMethod = config.optimMethod,
        state = T(
          "learningRate" -> config.learningRate,
          "weightDecay" -> config.weightDecay,
          "momentum" -> config.momentum,
          "dampening" -> 0.0,
          "learningRateSchedule" -> config.learningRateSchedule
        ),
        endWhen = config.endWhen
      )
      optimizer.setCache(param.cache + "/" + param.net, config.cacheTrigger)
      optimizer.setValidationTrigger(config.testTrigger)
      optimizer.addValidation(new Top1Accuracy[Float])
      optimizer.addValidation(new Top5Accuracy[Float])
      optimizer.overWriteCache()
      optimizer.optimize()
    })
  }

}
