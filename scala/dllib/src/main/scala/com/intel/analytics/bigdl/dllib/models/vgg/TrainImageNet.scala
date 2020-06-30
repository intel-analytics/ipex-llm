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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.optim.SGD.{Poly, SequentialSchedule, Warmup}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, MklBlas, MklDnn, OptimizerV1, OptimizerV2}
import com.intel.analytics.bigdl.visualization.TrainSummary
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).foreach(param => {
      val conf = Engine.createSparkConf().setAppName("Train VGG-16 on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val imageSize = 224
      val trainImageCounts = 1281167
      val batchSize = param.batchSize
      val folder = param.folder
      val classNumber = param.classNumber

      val trainDataSet = Utils.trainDataSet(folder + "/train", sc, imageSize, batchSize)
      val validateSet = Utils.valDataSet(folder + "/val", sc, imageSize, batchSize)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        Engine.getEngineType() match {
          case MklBlas =>
            Vgg_16(classNumber)
          case MklDnn =>
            nn.mkldnn.models.Vgg_16.graph(batchSize / Engine.nodeNumber(), classNumber)
        }
      }

      println(model)

      if (param.optimizerVersion.isDefined) {
        param.optimizerVersion.get.toLowerCase match {
          case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
          case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
        }
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[SGD[Float]]
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(trainImageCounts / batchSize).toInt
        val lrSchedules = SequentialSchedule(iterationsPerEpoch)

        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch.getOrElse(0)
        if (warmUpIteration != 0) {
          val delta = (param.maxLr - param.learningRate) / warmUpIteration
          lrSchedules.add(Warmup(delta), warmUpIteration)
          logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
            s"maxLr: ${param.maxLr}, delta: $delta")
        }

        lrSchedules.add(Poly(0.5, 40000), 40000 - warmUpIteration)

        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening,
          nesterov = param.nesterov, learningRateSchedule = lrSchedules)
      }

      val logdir = "vgg16-imagenet"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))

      val criterion = Engine.getEngineType() match {
        case MklBlas => CrossEntropyCriterion[Float]()
        case MklDnn => SoftmaxWithCriterion[Float]()
      }

      val optimizer = Optimizer(model, trainDataSet, criterion)
      val validationTrigger = Trigger.severalIteration(param.checkpointIteration)
      val validationMethods = Array(new Top1Accuracy[Float], new Top5Accuracy[Float])

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, validationTrigger)
      }

      optimizer
        .setGradientClippingByl2Norm(param.gradientL2NormThreshold.getOrElse(10000))
        .setOptimMethod(optimMethod)
        .setValidation(validationTrigger, validateSet, validationMethods)
        .setEndWhen(Trigger.severalIteration(param.maxIteration))
        .optimize()

      sc.stop()
    })
  }
}
