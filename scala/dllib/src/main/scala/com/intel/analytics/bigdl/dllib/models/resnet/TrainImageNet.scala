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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.inception.{ImageNet2012, ImageNet2012Val}
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.{BatchNormalization, Container, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def imageNetDecay(epoch: Int): Double = {
    if (epoch >= 80) {
      3
    } else if (epoch >= 60) {
      2
    } else if (epoch >= 30) {
      1
    } else {
      0.0
    }
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Train ResNet on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch, dataSet) =
        (224, DatasetType.ImageNet, param.nepochs, ImageNetDataSet)

      val trainDataSet = dataSet.trainDataSet(param.folder + "/train", sc, imageSize, batchSize)

      val validateSet = dataSet.valDataSet(param.folder + "/val", sc, imageSize, batchSize)

      val shortcut: ShortcutType = ShortcutType.B

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel =
          ResNet(classNum = param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth,
            "optnet" -> param.optnet, "dataSet" -> dataSetType))
        if (param.optnet) {
          ResNet.shareGradInput(curModel)
        }
        ResNet.modelInit(curModel)

        /* Here we set parallism specificall for BatchNormalization and its Sub Layers, this is
        very useful especially when you want to leverage more computing resources like you want
        to use as many cores as possible but you cannot set batch size too big for each core due
        to the memory limitation, so you can set batch size per core smaller, but the smaller
        batch size will increase the instability of convergence, the synchronization among BN
        layers basically do the parameters synchronization among cores and thus will avoid the
        instability while improves the performance a lot. */
        if (Engine.getEngineType() == MklBlas) setParallism(curModel, Engine.coreNumber)
        curModel
      }

      println(model)

      if (param.optimizerVersion.isDefined) {
        param.optimizerVersion.get.toLowerCase match {
          case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
          case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
        }
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        val optim = OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[SGD[Float]]
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
        optim
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration

        logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
          s"maxLr: $maxLr, " +
          s"delta: $delta, nesterov: ${param.nesterov}")
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening,
          nesterov = param.nesterov,
          learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      val logdir = "resnet-imagenet"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName)

      optimizer
        .setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
      sc.stop()
    })
  }

  private def setParallism(model: AbstractModule[_, _, Float], parallism: Int): Unit = {
    if (model.isInstanceOf[BatchNormalization[Float]]) {
      model.asInstanceOf[BatchNormalization[Float]].setParallism(parallism)
    }
    if(model.isInstanceOf[Container[_, _, Float]]) {
      model.asInstanceOf[Container[_, _, Float]].
        modules.foreach(sub => setParallism(sub, parallism))
    }
  }
}
