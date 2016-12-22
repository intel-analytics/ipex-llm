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
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.Logger
import org.apache.spark

import scala.util.Random

object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val sc = new SparkContext("local[1]", "train rnn")
      val dictionaryLength = 4000+1
      val wt = new WordTokenizer(
        param.folder + "/input.txt",
        param.folder,
        dictionaryLength = dictionaryLength-1,
        sc
      )
      wt.process()

      val dataArray = readSentence(param.folder + "/train.txt", dictionaryLength, sc)
      val labelSeq = readSentence(param.folder + "/label.txt", dictionaryLength, sc)

      val seq = Random.shuffle((1 to dataArray.length).toList)
      val seqTrain = seq.take(Math.floor(seq.length*0.8).toInt).toArray
      val seqVal   = seq.drop(Math.floor(seq.length*0.8).toInt).toArray

      val trainDataArray = seqTrain.collect(dataArray)
      val trainLabelArray = seqTrain.collect(labelSeq)
      val valDataArray = seqVal.collect(dataArray)
      val valLabelArray = seqVal.collect(labelSeq)

      logger.info("train: input size = " + trainDataArray.length)
      logger.info("train: label size = " + trainLabelArray.length)
      logger.info("test: input size = " + valDataArray.length)
      logger.info("test: label size = " + valLabelArray.length)

      val model = SimpleRNN(
        inputSize = dictionaryLength,
        hiddenSize = 40,
        outputSize = dictionaryLength,
        bpttTruncate = 4)
      model.reset()

      val state = T("learningRate" -> param.learningRate, "momentum" -> 0.0,
        "weightDecay" -> 0.0, "dampening" -> 0.0)

      val optimizer = new OptimizerSimpleRNN[Float](
        model = model,
        criterion = CrossEntropyCriterion[Float](),
        dictionaryLength = dictionaryLength
      )

      optimizer
        .setTrain(trainDataArray, trainLabelArray)
        .setValidation(Trigger.everyEpoch, valDataArray, valLabelArray)
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()


//      val (sampleInput, sampleLabel) = optimizer.convert(valDataArray(5), valLabelArray(5))
//      val sampleOutput = predict(model, sampleInput)
//      logger.info(s"sampleInput = ${valDataArray(5).mkString(",")}")
//      logger.info(s"sampleLabel = ${valLabelArray(5).mkString(",")}")
//      logger.info(s"sampleOutput = ${sampleOutput.mkString(",")}")

    })
  }
}
