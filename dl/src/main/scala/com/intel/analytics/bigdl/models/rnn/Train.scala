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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
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

      val (traintokens, valtokens, dictionary, maxTrainLength, maxValLength) = if (!sc.isDefined) {

        val tokens = SequencePreprocess(
          param.dataFolder + "/train.txt",
          param.sentFile,
          param.tokenFile).toStream

        val dictionary = Dictionary(tokens, param.vocabSize)
        dictionary.save(param.saveFolder)

        var maxTrainLength = tokens.map(x => x.length).max

        val valtokens = SequencePreprocess(
          param.dataFolder + "/val.txt",
          param.sentFile,
          param.tokenFile
        ).toStream
        var maxValLength = valtokens.map(x => x.length).max

        logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

        (DataSet.array(tokens.toArray), DataSet.array(valtokens.toArray),
          dictionary, maxTrainLength, maxValLength)
      } else {
        val tokens = SequencePreprocess(
          param.dataFolder + "/train.txt",
          sc = sc.get,
          param.sentFile,
          param.tokenFile)

        val dictionary = Dictionary(tokens, param.vocabSize)
        dictionary.save(param.saveFolder)

        var maxTrainLength = tokens.map(x => x.length).collect().max

        val valtokens = SequencePreprocess(
          param.dataFolder + "/val.txt",
          sc = sc.get,
          param.sentFile,
          param.tokenFile
        )
        var maxValLength = valtokens.map(x => x.length).collect().max

        logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

        (DataSet.rdd(tokens), DataSet.rdd(valtokens), dictionary, maxTrainLength, maxValLength)
      }

      val totalVocabLength = dictionary.vocabSize() + 1
      val startIdx = dictionary.getIndex("SENTENCESTART")
      val endIdx = dictionary.getIndex("SENTENCEEND")
      val padFeature = Tensor[Float]().resize(totalVocabLength)
      padFeature.setValue(endIdx + 1, 1.0f)
      val padLabel = startIdx

      val (trainSet, validationSet) = (
      traintokens
        .transform(TextToLabeledSentence(dictionary))
        .transform(LabeledSentenceToSample(totalVocabLength,
          Option(maxTrainLength), Option(maxTrainLength)))
        .transform(SampleToBatch(batchSize = param.batchSize)),
      valtokens
        .transform(TextToLabeledSentence(dictionary))
        .transform(LabeledSentenceToSample(totalVocabLength,
          Option(maxValLength), Option(maxValLength)))
        .transform(SampleToBatch(batchSize = param.batchSize)))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = totalVocabLength,
          hiddenSize = param.hiddenSize,
          outputSize = totalVocabLength)
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
        criterion = TimeDistributedCriterion[Float](CrossEntropyCriterion[Float]())
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
