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

package com.intel.analytics.bigdl.models.rnn

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.Logger


object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val inputDirect = new File(param.dataFolder)
      if (!inputDirect.isDirectory || inputDirect.list.length == 0) {
        throw new IllegalArgumentException(
          "data folder is not a directory or input files not exists!")
      }
      logger.info("preprocessing input text file ..")

      val wt = new WordTokenizer(
        param.dataFolder,
        param.saveFolder,
        dictionaryLength = param.vocabSize
      )
      wt.process()

      val dictionaryLength: Int = wt.length()
      logger.info("loading the training and testing data ..")
      val dataArray = loadData(param.saveFolder, dictionaryLength)
      val trainData = dataArray._1
      val valData = dataArray._2
      val trainMaxLength = dataArray._3
      val valMaxLegnth = dataArray._4

      val batchSize = 1

      val trainSet = DataSet.array(trainData)
           .transform(LabeledSentenceToSample(dictionaryLength))
           .transform(SampleToBatch(batchSize = batchSize))
      val validationSet = DataSet.array(valData)
           .transform(LabeledSentenceToSample(dictionaryLength))
           .transform(SampleToBatch(batchSize = batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = dictionaryLength,
          hiddenSize = param.hiddenSize,
          outputSize = dictionaryLength,
          bpttTruncate = param.bptt)
        curModel.reset()
        curModel
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T("learningRate" -> param.learningRate,
          "momentum" -> param.momentum,
          "weightDecay" -> param.weightDecay,
          "dampening" -> param.dampening)
      }

      Engine.init(1, param.coreNumber, false)
      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new CrossEntropyCriterion[Float](squeezeFlag = true)
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
    })
  }
}
