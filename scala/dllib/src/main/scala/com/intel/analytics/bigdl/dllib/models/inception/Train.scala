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
package com.intel.analytics.bigdl.models.inception

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim.SGD.{MultiStep, Poly, SequentialSchedule, Warmup}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, OptimizerV1, OptimizerV2, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainInceptionV1 {
  LoggerFilter.redirectSparkInfoLogs()


  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val imageSize = 224
      val conf = Engine.createSparkConf().setAppName("BigDL InceptionV1 Train Example")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val trainSet = ImageNet2012(
        param.folder + "/train",
        sc,
        imageSize,
        param.batchSize,
        Engine.nodeNumber(),
        Engine.coreNumber(),
        param.classNumber
      )
      val valSet = ImageNet2012Val(
        param.folder + "/val",
        sc,
        imageSize,
        param.batchSize,
        Engine.nodeNumber(),
        Engine.coreNumber(),
        param.classNumber
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else if (param.graphModel) {
        Inception_v1_NoAuxClassifier.graph(classNum = param.classNumber)
      } else {
        Inception_v1_NoAuxClassifier(classNum = param.classNumber)
      }

      val iterationPerEpoch = math.ceil(1281167.toDouble / param.batchSize).toInt
      val maxIteration = if (param.maxEpoch.isDefined) {
        iterationPerEpoch * param.maxEpoch.get
      } else param.maxIteration

      val warmupIteration = param.warmupEpoch.getOrElse(0) * iterationPerEpoch

      if (param.optimizerVersion.isDefined) {
        param.optimizerVersion.get.toLowerCase match {
          case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
          case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
        }
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        val warmupDelta = if (warmupIteration == 0) 0.0
          else (param.maxLr.getOrElse(param.learningRate) - param.learningRate) / warmupIteration
        val polyIteration = maxIteration - warmupIteration
        val lrSchedule = SequentialSchedule(iterationPerEpoch)
          .add(Warmup(warmupDelta), warmupIteration).add(Poly(0.5, maxIteration), polyIteration)
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = 0.9, dampening = 0.0, nesterov = false,
          learningRateSchedule = lrSchedule)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      val (checkpointTrigger, testTrigger, endTrigger) = if (param.maxEpoch.isDefined) {
        (Trigger.everyEpoch, Trigger.everyEpoch, Trigger.maxEpoch(param.maxEpoch.get))
      } else {
        (
          Trigger.severalIteration(param.checkpointIteration),
          Trigger.severalIteration(param.checkpointIteration),
          Trigger.maxIteration(param.maxIteration)
          )
      }

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, checkpointTrigger)
      }

      if (param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      if (param.gradientMin.isDefined && param.gradientMax.isDefined) {
        optimizer.setConstantGradientClipping(param.gradientMin.get.toFloat,
          param.gradientMax.get.toFloat)
      }

      if (param.gradientL2NormThreshold.isDefined) {
        optimizer.setGradientClippingByl2Norm(param.gradientL2NormThreshold.get.toFloat)
      }

      optimizer
        .setOptimMethod(optimMethod)
        .setValidation(testTrigger,
          valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(endTrigger)
        .optimize()
      sc.stop()
    })
  }
}
