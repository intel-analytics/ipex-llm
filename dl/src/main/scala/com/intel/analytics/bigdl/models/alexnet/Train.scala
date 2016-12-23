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

package com.intel.analytics.bigdl.models.alexnet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD.Regime
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Options._

  val imageSize = 224

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL AlexNet Train Example")
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
        1281167)
      val valSet = ImageNet2012(
        param.folder + "/val",
        sc,
        imageSize,
        param.batchSize,
        param.nodeNumber,
        param.coreNumber,
        param.classNumber,
        50000,
        trainSet
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        AlexNet_OWT(classNum = 1000)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> SGD.EpochSchedule(Array(
            Regime(1, 18, T("learningRate" -> 1e-2, "weightDecay" -> 2e-4)),
            Regime(19, 29, T("learningRate" -> 5e-3, "weightDecay" -> 2e-4)),
            Regime(30, 43, T("learningRate" -> 1e-3, "weightDecay" -> 0.0)),
            Regime(44, 52, T("learningRate" -> 5e-4, "weightDecay" -> 0.0)),
            Regime(53, 100, T("learningRate" -> 1e-4, "weightDecay" -> 0.0))
          ))
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new ClassNLLCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      optimizer
        .setState(state)
        .setValidation(Trigger.everyEpoch, valSet,
          Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(56))
        .optimize()
    })
  }
}
