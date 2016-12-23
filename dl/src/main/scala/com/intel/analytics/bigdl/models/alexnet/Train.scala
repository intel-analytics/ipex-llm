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

package com.intel.analytics.bigdl.models.alexnet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD.Regime
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

object Train {

  import Options._

  val batchSize = 128
  val imageSize = 224

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      Engine.setCoreNumber(param.coreNumber)

      val trainSet = ImageNet2012(Paths.get(param.folder, "train"), imageSize, batchSize, 1281167)
      val valSet = ImageNet2012(Paths.get(param.folder, "val"), imageSize, batchSize, 50000)

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
      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
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
