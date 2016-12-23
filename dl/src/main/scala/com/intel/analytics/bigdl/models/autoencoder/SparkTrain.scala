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

package com.intel.analytics.bigdl.models.autoencoder

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, MSECriterion, Module}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.DataSet
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._


object SparkTrain {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainSparkParser.parse(args, new TrainSparkParams()).map(param => {
      Engine.setCluster(param.nodesNumber, param.coreNumberPerNode)
      val batchSize = 150
      val maxEpoch = 10

      val conf = Engine.sparkConf()
        .setAppName("Train Autoencoder on MNIST")
        .set("spark.akka.frameSize", 64.toString)
      val sc = new SparkContext(conf)

      val trainData = Paths.get(param.folder, "/train-images.idx3-ubyte")
      val trainLabel = Paths.get(param.folder, "/train-labels.idx1-ubyte")

      val trainDataSet = DataSet.array(load(trainData, trainLabel), sc, param.nodesNumber)
        .transform(SampleToGreyImg(28, 28))
        .transform(GreyImgNormalizer(trainMean, trainStd))
        .transform(GreyImgToAEBatch(batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        AE(classNum = 32)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> 0.01,
          "weightDecay" -> 0.0005,
          "momentum" -> 0.9,
          "dampening" -> 0.0
        )
      }

      val optimizer = new DistriOptimizer[Float](
        model = model,
        dataset = trainDataSet,
        criterion = new MSECriterion[Float]()
      )

      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }
      optimizer
        .setState(state)
        .setOptimMethod(new Adagrad[Float]())
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
    })
  }
}

