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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, OptimizerV1, OptimizerV2, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

object TrainCIFAR10 {
  LoggerFilter.redirectSparkInfoLogs()


  import Utils._

  def cifar10Decay(epoch: Int): Double =
    if (epoch >= 122) 2.0 else if (epoch >= 81) 1.0 else 0.0

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Train ResNet on Cifar10")
        // Will throw exception without this config when has only one executor
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = param.batchSize
      val (imageSize, lrSchedule, maxEpoch, dataSet) =
        (32, DatasetType.CIFAR10, param.nepochs, Cifar10DataSet)

      val trainDataSet = dataSet.trainDataSet(param.folder, sc, imageSize, batchSize)

      val validateSet = dataSet.valDataSet(param.folder, sc, imageSize, batchSize)

      val shortcut: ShortcutType = param.shortcutType match {
        case "A" => ShortcutType.A
        case "B" => ShortcutType.B
        case _ => ShortcutType.C
      }

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = if (param.graphModel) {
          ResNet.graph(param.classes,
            T("shortcutType" -> shortcut, "depth" -> param.depth, "optnet" -> param.optnet))
        } else {
          ResNet(param.classes,
            T("shortcutType" -> shortcut, "depth" -> param.depth, "optnet" -> param.optnet))
        }
        if (param.optnet) {
          ResNet.shareGradInput(curModel)
        }
        ResNet.modelInit(curModel)
        curModel
      }

      if (param.optimizerVersion.isDefined) {
        param.optimizerVersion.get.toLowerCase match {
          case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
          case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
        }
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening,
          nesterov = param.nesterov, learningRateSchedule = SGD.EpochDecay(cifar10Decay))
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
      sc.stop()

    })
  }
}
