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

import java.io.{BufferedReader, FileReader}
import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

import scala.collection.mutable.ArrayBuffer

object Train {

  import Utils._

  def readNumbers(filename: String, dictionarySize: Int): ArrayBuffer[Array[Int]] = {
    val inputBuffer = ArrayBuffer[Array[Int]]()
    val br = new BufferedReader(new FileReader(filename))
    var line = br.readLine()
    var sampleIndex = 1
    while (line != null) {
      sampleIndex += 1
      line = line.replaceAll("[^\\d,]", "")
      val stringArray = line.split(",")
      val curInput = new Array[Int](stringArray.length)
      for ((index, i) <- stringArray.zipWithIndex) {
        curInput(i) = index.toInt + 1
      }
      inputBuffer.append(curInput)
      line = br.readLine()
    }
    br.close()
    inputBuffer
  }


  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val dictionaryLength = 4000
      val inputTrainArray = readNumbers(param.folder + "/train.bn", dictionaryLength)
      val trainLabelArray = readNumbers(param.folder + "/trainLabel.bn", dictionaryLength)

      val inputTestArray = readNumbers(param.folder + "/test.bn", dictionaryLength)
      val testLabelArray = readNumbers(param.folder + "/testLabel.bn", dictionaryLength)

      println("train: input size = " + inputTrainArray.length)
      println("train: label size = " + trainLabelArray.length)
      println("test: input size = " + inputTestArray.length)
      println("test: label size = " + testLabelArray.length)

      val model = SimpleRNN(
        dictionaryLength,
        40,
        dictionaryLength,
        4)
      model.reset()

      val state = T("learningRate" -> param.learningRate, "momentum" -> 0.0,
        "weightDecay" -> 0.0, "dampening" -> 0.0)

      Engine.setCoreNumber(param.coreNumber)



      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> param.learningRate
        )
      }

      val optimizer = new LocalOptimizer[Float](
        model = model,
        dataset = trainSet.transform(normalizer).transform(GreyImgToBatch(param.batchSize)),
        criterion = new ClassNLLCriterion[Float]()
      )
      if (param.cache.isDefined) {
        optimizer.setCache(param.cache.get, Trigger.everyEpoch)
      }

      val validationSet = DataSet.array(load(validationData, validationLabel))
        .transform(SampleToGreyImg(28, 28))
      val normalizerVal = GreyImgNormalizer(validationSet)
      val valSet = validationSet.transform(normalizerVal)
        .transform(GreyImgToBatch(param.batchSize))

      optimizer
        .setValidation(Trigger.everyEpoch, valSet, Array(new Top1Accuracy[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
