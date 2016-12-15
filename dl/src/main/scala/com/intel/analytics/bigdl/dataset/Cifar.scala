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

import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.ImageNetLocal.Config
import com.intel.analytics.bigdl.models.ResNet
import com.intel.analytics.bigdl.models.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.cifar.VggLike
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD.{EpochDecay, EpochStep}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.T
import scopt.OptionParser

object Cifar10Local {
  case class Cifar10LocalParam(
    folder: String = "/home/ywan/Documents/data/Cifar-10",
    net: String = "resnet",
    optnet: Boolean = true
  )

  case class Config(
    model: Module[Float],
    criterion: Criterion[Float],
    optimMethod: OptimMethod[Float],
    imageSize: Int,
    momentum: Double,
    weightDecay: Double,
    endWhen: Trigger,
    learningRate: Double)
  //  learningRateSchedule: LearningRateSchedule)

  private val configs = Map(
    "vgg" -> Config(
      VggLike(classNum = 10),
      ClassNLLCriterion[Float](),
      new SGD[Float](),
      imageSize = 32,
      momentum = 0.9,
      weightDecay = 0.0005,
      endWhen = Trigger.maxEpoch(90),
      learningRate = 0.1),
      //learningRateSchedule = SGD.Step(100000, 0.1)),
    "resnet" -> Config(
      ResNet[Float](classNum = 10, T("shortcutType" -> ShortcutType.A, "depth" -> 20, "dataset" -> DatasetType.CIFAR10))
        .asInstanceOf[Module[Float]],
      CrossEntropyCriterion[Float](),
      new SGD[Float](),
      imageSize = 32,
      momentum = 0.9,
      weightDecay = 1e-4,
      endWhen = Trigger.maxEpoch(164),
      learningRate = 0.1))
    //  learningRateSchedule = SGD.EpochDecay())
  //)

  private val parser = new OptionParser[Cifar10LocalParam]("Spark-DL Cifar10 Local Example") {
    head("Spark-DL Cifar10 Local Example")
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[Boolean]("optnet")
      .text("share several tensors to reduce memory usage")
      .action((x, c) => c.copy(optnet = x))
    opt[String]('n', "net")
      .text("net type : vgg | resnet")
      .action((x, c) => c.copy(net = x.toLowerCase))
      .validate(v =>
        if (Set("vgg", "resnet").contains(v.toLowerCase())) {
          success
        } else {
          failure("Net type can only be vgg |resnet in this example")
        }
      )
  }

  def main(args: Array[String]) {
    parser.parse(args, new Cifar10LocalParam()).map(param => {
      val config = configs(param.net)
      val trainDataSource = new CifarDataSource(Paths.get(param.folder + "/train"), looped = true)
      val validationDataSource = new CifarDataSource(Paths.get(param.folder + "/val"),
        looped = false)

      //val cropper = RGBImageCropper(cropWidth = config.imageSize, cropHeight = config.imageSize)
      val randomCropper = RGBImageRandomCropper(cropWidth = config.imageSize, cropHeight = config.imageSize, padding = 4)
      val flipper = HFlip(0.5)

      val arrayToImage = ArrayByteToRGBImage()
      val normalizer = RGBImageNormalizer(trainDataSource -> arrayToImage) //125.3, 123.0, 113.9, 63.0, 62.1, 66.7)
      val toTensor = new RGBImageToTensor(batchSize = 100)


      if (param.net.equals("resnet")) {
        println(s"model is ${param.net}, initializing the model")
        ResNet.modelInit(config.model)
      }
      if (param.optnet) {
        println(s"model is ${param.net}, setting shared variable")
        ResNet.shareGradInput(config.model)
      }

      val optimizer = new LocalOptimizer[Float](
        //data = trainDataSource -> arrayToImage -> normalizer -> toTensor,
        //validationData = validationDataSource -> arrayToImage -> normalizer -> toTensor,
        data = trainDataSource -> arrayToImage -> normalizer -> flipper -> randomCropper -> toTensor,
        validationData = validationDataSource -> arrayToImage -> normalizer -> toTensor,
        model = config.model, //VggLike[Float](classNum = 10),
        criterion = config.criterion, //new ClassNLLCriterion[Float](),
        optimMethod = config.optimMethod, //new SGD[Float](),
        state = T(
          "learningRate" -> config.learningRate, //0.01,
          "weightDecay" -> config.weightDecay, //0.0005,
          "momentum" -> config.momentum, //0.9,
          "dampening" -> config.momentum, //0.0,
          "learningRateSchedule" -> EpochDecay(DatasetType.CIFAR10) //EpochStep(25, 0.5)
        ),
        endWhen = Trigger.maxEpoch(164)
      )
      optimizer.setValidationTrigger(Trigger.everyEpoch)
      optimizer.addValidation(new Top1Accuracy[Float])

      optimizer.optimize()
    })
  }
}
