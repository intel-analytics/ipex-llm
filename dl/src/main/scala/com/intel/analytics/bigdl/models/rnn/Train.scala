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

package com.intel.analytics.bigdl.models.rnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.{TensorSeqToBatch, TextSeqToTensorSeq}
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.Logger

object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val sc = new SparkContext("local[1]", "train rnn")
      val dictionaryLength = 4000 + 1
      val wt = new WordTokenizer(
        param.folder + "/input.txt",
        param.folder,
        dictionaryLength = dictionaryLength-1,
        sc
      )
      wt.process()

      val dataArray = readSentence(param.folder, dictionaryLength, sc)
      val trainData = dataArray._1
      val valData = dataArray._2

      val trainSet = DataSet.array(trainData)
        .transform(TextSeqToTensorSeq(dictionaryLength))
        .transform(TensorSeqToBatch())
      val validationSet = DataSet.array(valData)
        .transform(TextSeqToTensorSeq(dictionaryLength))
        .transform(TensorSeqToBatch())

      val model = SimpleRNN(
        inputSize = dictionaryLength,
        hiddenSize = 40,
        outputSize = dictionaryLength,
        bpttTruncate = 4)
      model.reset()

      val state = T("learningRate" -> param.learningRate, "momentum" -> 0.0,
        "weightDecay" -> 0.0, "dampening" -> 0.0)


      Engine.setCoreNumber(1)
      Engine.model.setPoolSize(param.coreNumber)
      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new CrossEntropyCriterion[Float]()
      )

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
