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
package com.intel.analytics.bigdl.models.inception

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object GoogleNetv1_SparkTrain {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val batchSize = param.batchSize.getOrElse(1568)
      val imageSize = 224
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL Inception v1 Train Example")
            .set("spark.task.maxFailures", "1")
          new SparkContext(conf)
        })

      val trainSet = ImageNet2012(
        param.folder + "/train",
        sc,
        imageSize,
        batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber
      )
      val valSet = ImageNet2012(
        param.folder + "/val",
        sc,
        imageSize,
        batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        GoogleNet_v1_NoAuxClassifier(classNum = param.classNumber)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.0898,
          "weightDecay" -> 0.0001,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> SGD.Poly(0.5, 62000)
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCache(param.checkpoint.get, Trigger.severalIteration(620))
      }

      optimizer
        .setState(state)
        .setValidation(Trigger.severalIteration(620),
          valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxIteration(62000))
        .optimize()
    })
  }
}

object GoogleNetv2_SparkTrain {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val batchSize = param.batchSize.getOrElse(1344)
      val imageSize = 224
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL Inception v1 Train Example")
            .set("spark.task.maxFailures", "1")
          new SparkContext(conf)
        })
      val trainSet = ImageNet2012(
        param.folder + "/train",
        sc,
        imageSize,
        batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber
      )
      val valSet = ImageNet2012(
        param.folder + "/val",
        sc,
        imageSize,
        batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        GoogleNet_v2_NoAuxClassifier(classNum = param.classNumber)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.1,
          "weightDecay" -> 0.0002,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> SGD.Step(900, 0.96)
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCache(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setState(state)
        .setValidation(Trigger.everyEpoch,
          valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(100))
        .optimize()
    })
  }
}
