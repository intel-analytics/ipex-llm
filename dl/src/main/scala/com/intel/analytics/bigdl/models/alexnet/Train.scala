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

import com.intel.analytics.bigdl.nn.{Module, ClassNLLCriterion}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

object Train {

  object Local {

    import Options._

    def main(args: Array[String]): Unit = {
      trainLocalParser.parse(args, new TrainLocalParams()).map(param => {
        val batchSize = 256
        val imageSize = 227

        val trainData = Paths.get(param.folder, "train")
        val trainDataSet = ImageNet2012(trainData, imageSize, batchSize, param.coreNumber)
        val validationData = Paths.get(param.folder, "val")
        val validateDataSet = ImageNet2012(validationData, imageSize, batchSize, param.coreNumber)

        val model = if (param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          AlexNet(classNum = 1000)
        }

        val state = if (param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> 0.01,
            "weightDecay" -> 0.0005,
            "momentum" -> 0.9,
            "dampening" -> 0.0,
            "learningRateSchedule" -> SGD.Step(100000, 0.1)
          )
        }

        Engine.setCoreNumber(param.coreNumber)
        val optimizer = new LocalOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]()
        )
        if (param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.severalIteration(10000))
        }
        optimizer
          .setState(state)
          .setValidation(Trigger.severalIteration(10000),
            validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
          .setEndWhen(Trigger.maxIteration(450000))
          .optimize()
      })
    }
  }

}
