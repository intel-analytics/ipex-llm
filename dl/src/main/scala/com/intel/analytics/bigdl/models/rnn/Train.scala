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

package com.intel.analytics.bigdl.models.rnn

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

import scala.io.Source

object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark").map(conf => {
        conf.setAppName("Train rnn on text")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val (trainSet, validationSet, dictionaryLength) = if (!sc.isDefined) {
        val logData = Source.fromFile(param.dataFolder + "/" + "train.txt").getLines().toArray
        val trainSents = DataSet.array(logData
          .filter(!_.isEmpty)).transform(SentenceSplitter())
        val output_sents = trainSents.toLocal().data(train = false).flatMap(item => item.iterator)
        val tokens = DataSet.array(output_sents.toArray).transform(DocumentTokenizer())
        val dictionary = Dictionary(tokens.toLocal().data(false), param.vocabSize)
        dictionary.save(param.saveFolder)
        println("vocabulary size = " + dictionary.vocabSize())
        var maxTrainLength = 0
        tokens.toLocal().data(false).foreach(x => {
          maxTrainLength = math.max(maxTrainLength, x.length)
        })
        val valData = Source.fromFile(param.dataFolder + "/" + "val.txt").getLines().toArray
        val valSents = DataSet.array(valData
          .filter(!_.isEmpty)).transform(SentenceSplitter())
        val valoutput = valSents.toLocal().data(train = false).flatMap(item => item.iterator)
        val valtokens = DataSet.array(valoutput.toArray).transform(DocumentTokenizer())
        var maxValLength = 0
        valtokens.toLocal().data(false).foreach(x => {
          maxValLength = math.max(maxValLength, x.length)
        })

        println(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")
        (tokens
          .transform(TextToLabeledSentence(dictionary))
          .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1,
            Option(maxTrainLength), Option(maxTrainLength)))
          .transform(SampleToBatch(batchSize = param.batchSize)),
          valtokens
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1,
              Option(maxTrainLength), Option(maxTrainLength)))
            .transform(SampleToBatch(batchSize = param.batchSize)),
          dictionary.vocabSize() + 1)
      } else {
        val trainSents = DataSet.rdd(sc.get.textFile(param.dataFolder + "/" + "train.txt")
          .filter(!_.isEmpty)).transform(SentenceSplitter())
        val output_sents = trainSents.toDistributed().data(train = false).collect().flatten
        val tokens = DataSet.rdd(sc.get.parallelize(output_sents)).transform(DocumentTokenizer())
        val dictionary = Dictionary(tokens.toDistributed().data(false),
          param.vocabSize)
        dictionary.save(param.saveFolder)
        val valSents = DataSet.rdd(sc.get.textFile(param.dataFolder + "/" + "val.txt")
          .filter(!_.isEmpty)).transform(SentenceSplitter())
        val valoutput = valSents.toDistributed().data(train = false).collect().flatten
        val valtokens = DataSet.rdd(sc.get.parallelize(valoutput)).transform(DocumentTokenizer())
        (tokens
          .transform(TextToLabeledSentence(dictionary))
          .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
          .transform(SampleToBatch(batchSize = param.batchSize)),
          valtokens
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
            .transform(SampleToBatch(batchSize = param.batchSize)),
          dictionary.vocabSize() + 1)
      }

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = dictionaryLength,
          hiddenSize = param.hiddenSize,
          outputSize = dictionaryLength)
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

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = CrossEntropyCriterion[Float]()
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
