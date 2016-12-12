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
package com.intel.analytics.bigdl.models.googlenet

import java.nio.file.Paths

import com.intel.analytics.bigdl.models.alexnet.{AlexNet, DataSet, Options}
import com.intel.analytics.bigdl.models.alexnet.Options._
import com.intel.analytics.bigdl.nn.{Criterion, ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, Activities, T}
import org.apache.spark.SparkContext

object Train {
  object GoogleNetv1_Local {
    import Options._
    def main(args: Array[String]): Unit = {
      trainLocalParser.parse(args, new TrainLocalParams()).map(param => {
        val batchSize = 32
        val imageSize = 224

        val trainData = Paths.get(param.folder, "train")
        val trainDataSet = DataSet.localDataSet(trainData, imageSize, batchSize,
          param.coreNumber, true)
        val validationData = Paths.get(param.folder, "val")
        val validateDataSet = DataSet.localDataSet(validationData, imageSize, batchSize,
          param.coreNumber, false)

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          GoogleNet_v1_NoAuxClassifier(classNum = 1000)
        }

        val state = if(param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> 0.01,
            "weightDecay" -> 0.0002,
            "momentum" -> 0.9,
            "dampening" -> 0.0,
            "learningRateSchedule" -> SGD.Poly(0.5, 2400000)
          )
        }

        val optimizer = new LocalOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Activities, Float]],
          coreNumber = param.coreNumber
        )
        if(param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.everyEpoch)
        }
        optimizer
          .setState(state)
          .setValidation(Trigger.everyEpoch,
            validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
          .setEndWhen(Trigger.maxIteration(2400000))
          .optimize()
      })
    }
  }

  object GoogleNetv1_Spark {
    import Options._
    def main(args: Array[String]): Unit = {
      trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
        val batchSize = 1600
        val imageSize = 224

        val conf = Engine.sparkConf().setAppName("BigDL GoogleNet v1 Example")
        val sc = new SparkContext(conf)
        val trainDataSet = DataSet.DistriDataSet(
          param.folder + "/train",
          sc,
          imageSize,
          batchSize,
          param.nodesNumber,
          param.coreNumberPerNode,
          true)
        val validateDataSet = DataSet.DistriDataSet(
          param.folder + "/val",
          sc,
          imageSize,
          batchSize,
          param.nodesNumber,
          param.coreNumberPerNode,
          false)

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          GoogleNet_v1_NoAuxClassifier(classNum = 1000)
        }

        val state = if(param.stateSnapshot.isDefined) {
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

        val optimizer = new DistriOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Activities, Float]],
          nodeNumber = param.nodesNumber,
          coresPerNode = param.coreNumberPerNode
        )

        if(param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.everyEpoch)
        }

        optimizer
          .setState(state)
          .setValidation(Trigger.everyEpoch,
            validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
          .setEndWhen(Trigger.maxIteration(62000))
          .optimize()
      })
    }
  }
}

