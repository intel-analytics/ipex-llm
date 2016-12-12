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

package com.intel.analytics.bigdl.models.lenet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{Module, Criterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Activities, Engine, MklBlas, T}
import org.apache.spark.{SparkConf, SparkContext}

object Train {
  object Local {
    import Options._
    def main(args: Array[String]): Unit = {
      trainLocalParser.parse(args, new TrainLocalParams()).map(param => {
        val trainData = Paths.get(param.folder, "/train-images.idx3-ubyte")
        val trainDLabel = Paths.get(param.folder, "/train-labels.idx1-ubyte")
        val validationData = Paths.get(param.folder, "/t10k-images.idx3-ubyte")
        val validationLabel = Paths.get(param.folder, "/t10k-labels.idx1-ubyte")

        val trainDataSet = DataSet.localDataSet(trainData, trainDLabel, true, param.batchSize)
        import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          LeNet5(classNum = 10)
        }
        val state = if(param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> param.learningRate
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
        val validateDataSet = DataSet.localDataSet(validationData, validationLabel, false,
          param.batchSize)
        optimizer
          .setValidation(Trigger.everyEpoch, validateDataSet, Array(new Top1Accuracy[Float]))
          .setState(state)
          .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
          .optimize()
      })
    }
  }

  object Spark {
    import Options._
    def main(args: Array[String]): Unit = {
      trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
        val trainData = Paths.get(param.folder, "/train-images.idx3-ubyte")
        val trainDLabel = Paths.get(param.folder, "/train-labels.idx1-ubyte")
        val validationData = Paths.get(param.folder, "/t10k-images.idx3-ubyte")
        val validationLabel = Paths.get(param.folder, "/t10k-labels.idx1-ubyte")

        val conf = Engine.sparkConf().setAppName("Train Lenet")
        val sc = new SparkContext(conf)

        val trainDataSet = DataSet.distributedDataSet(trainData, trainDLabel, true, sc,
          param.nodesNumber, param.batchSize)
        import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

        val model = if(param.modelSnapshot.isDefined) {
          Module.load[Float](param.modelSnapshot.get)
        } else {
          LeNet5(classNum = 10)
        }
        val state = if(param.stateSnapshot.isDefined) {
          T.load(param.stateSnapshot.get)
        } else {
          T(
            "learningRate" -> param.learningRate
          )
        }

        val optimizer = new DistriOptimizer[Float](
          model = model,
          dataset = trainDataSet,
          criterion = new ClassNLLCriterion[Float]().asInstanceOf[Criterion[Activities, Float]],
          nodeNumber = param.nodesNumber,
          coresPerNode = param.coreNumberPerNode
        )
        val validateDataSet = DataSet.distributedDataSet(validationData, validationLabel, false,
          sc, param.nodesNumber, param.batchSize)
        if(param.cache.isDefined) {
          optimizer.setCache(param.cache.get, Trigger.everyEpoch)
        }
        optimizer
          .setValidation(Trigger.everyEpoch, validateDataSet, Array(new Top1Accuracy[Float]))
          .setState(state)
          .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
          .optimize()
      })
    }
  }
}
