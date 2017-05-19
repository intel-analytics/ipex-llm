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

package com.intel.analytics.bigdl.example.numerals_classification

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim.SGD.{EpochStep, Step}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkContext}

object Train {

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL classification Train Example")
            .set("spark.task.maxFailures", "1")
          new SparkContext(conf)
        })

      val trainData = Paths.get(param.folder, "/train.bin")
      val trainLabel = Paths.get(param.folder, "/train_label.bin")
      val validationData = Paths.get(param.folder, "/test.bin")
      val validationLabel = Paths.get(param.folder, "/test_label.bin")

      val trainDataSet = (if (sc.isDefined) {
        DataSet.array(load(trainData, trainLabel), sc.get)
      } else {
        DataSet.array(load(trainData, trainLabel))
      }).transform(SampleToBatch(param.batchSize, 4))

      val validateDataSet = (if (sc.isDefined) {
        DataSet.array(load(validationData, validationLabel), sc.get)
      } else {
        DataSet.array(load(validationData, validationLabel))
      }).transform(SampleToBatch(param.batchSize, 4))

      import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        Numerals_Classification(classNum = 2)
      }
      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> param.learningRate,
          "weightDecay" -> 0.0005,
          "momentum" -> 0.9,
          "learningRateSchedule" -> EpochStep(6, 0.1)
        )
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new ClassNLLCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      optimizer
        .setValidation(Trigger.everyEpoch, validateDataSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
