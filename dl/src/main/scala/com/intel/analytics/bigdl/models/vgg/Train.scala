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
package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.{Module, ClassNLLCriterion}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

object Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark").map(conf => {
        conf.setAppName("Train Vgg on Cifar10")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val trainDataSet = (if (sc.isDefined) {
        DataSet.array(Utils.loadTrain(param.folder), sc.get)
      } else {
        DataSet.array(Utils.loadTrain(param.folder))
      }) -> BytesToBGRImg() -> BGRImgNormalizer(trainMean, trainStd) -> BGRImgToBatch(
        param.batchSize)

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
          "hyperParameterScheduler" -> SGD.EpochStep[Float](25, 0.5)
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new ClassNLLCriterion[Float]()
      )

      val validateSet = (if (sc.isDefined) {
        DataSet.array(Utils.loadTest(param.folder), sc.get)
      } else {
        DataSet.array(Utils.loadTest(param.folder))
      }) -> BytesToBGRImg() -> BGRImgNormalizer(testMean, testStd) -> BGRImgToBatch(param
        .batchSize)

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }
      optimizer
        .setValidation(Trigger.everyEpoch, validateSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
