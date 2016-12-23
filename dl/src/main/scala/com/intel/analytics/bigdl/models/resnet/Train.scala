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

package com.intel.analytics.bigdl.models.resnet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import Options._
import com.intel.analytics.bigdl.models.resnet.Options.{TrainSparkParams => _, trainSparkParser => _, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

object LocalTrain {

  def imageNetDecay(epoch: Int): Double = math.floor((epoch - 1) / 30)
  def cifar10Decay(epoch: Int): Double =
    if (epoch >= 122) 2.0 else if (epoch >= 81) 1.0 else 0.0


  def main(args: Array[String]): Unit = {
    trainLocalParser.parse(args, new TrainLocalParams()).map(param => {

      val batchSize = 32
      val (imageSize, lrSchedule, maxEpoch, dataSet) = param.dataset match {
        // case "imagenet" => (224, DatasetType.ImageNet, 90, ImagenetDataSet)
        case _ => (32, DatasetType.CIFAR10, 165, Cifar10DataSet)
      }

      val trainData = Paths.get(param.folder, "train")
      val trainDataSet = dataSet.localTrainDataSet(trainData, batchSize, imageSize)
      val validationData = Paths.get(param.folder, "val")
      val validateDataSet = dataSet.localValDataSet(validationData, batchSize, imageSize)


      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = param.dataset match {
        case "imagenet" =>
          ResNet(classNum = 100, T("shortcutType" -> ShortcutType.B, "depth" -> 18))
        case _ =>
          ResNet(classNum = 10, T("shortcutType" -> ShortcutType.A, "depth" -> 20))
        }
        ResNet.shareGradInput(curModel)
        ResNet.modelInit(curModel)
        curModel
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.1,
          "weightDecay" -> 1e-4,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "nesterov" -> true,
          "learningRateSchedule" -> SGD.EpochDecay(cifar10Decay)
        )
      }

      Engine.setCoreNumber(param.coreNumber)
      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }

      optimizer
        .setState(state)
        .setValidation(Trigger.everyEpoch,
          validateDataSet, Array(new Top1Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()

    })
  }
}

object SparkTrain {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.DEBUG)

  def imageNetDecay(epoch: Int): Double = math.floor((epoch - 1) / 30)
  def cifar10Decay(epoch: Int): Double =
    if (epoch >= 122) 2.0 else if (epoch >= 81) 1.0 else 0.0

  def main(args: Array[String]): Unit = {
    trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
      val batchSize = 128
      val (imageSize, lrSchedule, maxEpoch, dataSet) = param.dataset match {
        // case "imagenet" => (224, DatasetType.ImageNet, 90, ImagenetDataSet)
        case _ => (32, DatasetType.CIFAR10, 165, Cifar10DataSet)
      }

      Engine.setCluster(param.nodesNumber, param.coreNumberPerNode)
      val conf = Engine.sparkConf()
        .setAppName("Train ResNet on Cifar10")
        .set("spark.akka.frameSize", 64.toString)

      val sc = new SparkContext(conf)

      val trainData = Paths.get(param.folder, "train")
      val trainDataSet =
        dataSet.distributedTrainDataSet(trainData, sc, param.nodesNumber, imageSize, batchSize)
      val validationData = Paths.get(param.folder, "val")
      val validateDataSet =
        dataSet.distributedValDataSet(validationData, sc, param.nodesNumber, imageSize, batchSize)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = param.dataset match {
          case "imagenet" =>
            ResNet(classNum = 100, T("shortcutType" -> ShortcutType.B, "depth" -> 18))
          case _ =>
            ResNet(classNum = 10, T("shortcutType" -> ShortcutType.A, "depth" -> 20))
        }
        ResNet.shareGradInput(curModel)
        ResNet.modelInit(curModel)
        curModel
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.1,
          "weightDecay" -> 1e-4,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "nesterov" -> true,
          "learningRateSchedule" -> SGD.EpochDecay(cifar10Decay)
        )
      }


      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )

      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }
      optimizer
        .setValidation(Trigger.everyEpoch,
          validateDataSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
    })
  }
}
