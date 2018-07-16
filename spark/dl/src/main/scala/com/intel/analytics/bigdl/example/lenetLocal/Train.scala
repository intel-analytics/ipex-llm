/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.example.lenetLocal

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}


object Train {
  LoggerFilter.redirectSparkInfoLogs()


  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      System.setProperty("bigdl.disable.mklBlockTime", "true");
      System.setProperty("bigdl.localMode", "true")
      System.setProperty("bigdl.coreNumber", "1") // param.coreNumber.toString)
      System.setProperty("bigdl.engineType", "mkldnn")
      val coreNumber: Int = System.getProperty("bigdl.mklNumThreads",
        s"${Runtime.getRuntime.availableProcessors() / 2}").toInt
      Engine.setCoreNumber(1)
      MklDnn.setNumThreads(coreNumber)

      Engine.init

      val trainData = param.folder + "/train-images-idx3-ubyte"
      val trainLabel = param.folder + "/train-labels-idx1-ubyte"
      val validationData = param.folder + "/t10k-images-idx3-ubyte"
      val validationLabel = param.folder + "/t10k-labels-idx1-ubyte"

      val (model, inputFormats) = if (param.modelSnapshot.isDefined) {
        (Module.load[Float](param.modelSnapshot.get), null)
      } else {
//        LeNet5(classNum = 10)
        LeNet5.dnn(classNum = 10, batchSize = param.batchSize)
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate,
          learningRateDecay = param.learningRateDecay)
      }

      val trainSet = DataSet.array(load(trainData, trainLabel)) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = CrossEntropyCriterion[Float]())
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      if (param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      val validationSet = DataSet.array(load(validationData, validationLabel)) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)

      optimizer
        .setValidation(
          trigger = Trigger.everyEpoch,
          dataset = validationSet,
          vMethods = Array(new Top1Accuracy, new Top5Accuracy[Float], new Loss[Float]))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
