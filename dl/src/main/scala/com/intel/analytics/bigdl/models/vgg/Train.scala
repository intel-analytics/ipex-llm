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
package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.{Module, ClassNLLCriterion}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.DataSet
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._


object LocalTrain {
  import Utils._

  def main(args: Array[String]): Unit = {
    trainLocalParser.parse(args, new TrainLocalParams()).map(param => {
      val batchSize = 128
      val maxEpoch = 90

      val trainSet = DataSet.ImageFolder.images(Paths.get(param.folder, "/train"), 32)
        .transform(BGRImgNormalizer(trainMean, trainStd))
        .transform(BGRImgToBatch(batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        VggForCifar10(classNum = 10)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.01,
          "weightDecay" -> 0.0005,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> SGD.EpochStep(25, 0.5)
        )
      }

      Engine.setCoreNumber(param.coreNumber)
      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }

      val validationSet = DataSet.ImageFolder.images(Paths.get(param.folder, "/val"), 32)
        .transform(BGRImgNormalizer(testMean, testStd))
        .transform(BGRImgToBatch(batchSize))

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
    })
  }
}

object SparkTrain {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)


  import Utils._

  def main(args: Array[String]): Unit = {
    trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
      Engine.setCluster(param.nodesNumber, param.coreNumberPerNode)
      val batchSize = param.batchSize.getOrElse(128)
      val maxEpoch = 90

      val conf = Engine.sparkConf()
        .setAppName("Train Vgg on Cifar10")
        .set("spark.akka.frameSize", 64.toString)
      val sc = new SparkContext(conf)

      val trainDataSet = DataSet.ImageFolder
        .images(Paths.get(param.folder, "/train"), sc, param.nodesNumber, 32)
        .transform(BGRImgNormalizer(trainMean, trainStd))
        .transform(BGRImgToBatch(batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        VggForCifar10(classNum = 10)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.01,
          "weightDecay" -> 0.0005,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> SGD.EpochStep(25, 0.5)
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      val validateSet = DataSet.ImageFolder
        .images(Paths.get(param.folder, "/val"), sc, param.nodesNumber, 32)
        .transform(BGRImgNormalizer(testMean, testStd))
        .transform(BGRImgToBatch(batchSize))

      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }
      optimizer
        .setValidation(Trigger.everyEpoch, validateSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
    })
  }
}
