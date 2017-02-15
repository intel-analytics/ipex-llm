/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
import com.intel.analytics.bigdl.utils.{Engine, MklDnn, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainInceptionV1 {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
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
        param.batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber,
        1281167
      )
      val valSet = ImageNet2012(
        param.folder + "/val",
        sc,
        imageSize,
        param.batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber,
        50000
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        Inception_v1_NoAuxClassifier(classNum = param.classNumber)
      }

      if (Engine.getEngineType == MklDnn) {
        model.convertToMklDnn()
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> param.learningRate,
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

      val (checkpointTrigger, testTrigger, endTrigger) = if (param.maxEpoch.isDefined) {
        (Trigger.everyEpoch, Trigger.everyEpoch, Trigger.maxEpoch(param.maxEpoch.get))
      } else {
        (
          Trigger.severalIteration(620),
          Trigger.severalIteration(620),
          Trigger.maxIteration(param.maxIteration)
          )
      }

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, checkpointTrigger)
      }

      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setState(state)
        .setValidation(testTrigger,
          valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(endTrigger)
        .optimize()
    })
  }
}
