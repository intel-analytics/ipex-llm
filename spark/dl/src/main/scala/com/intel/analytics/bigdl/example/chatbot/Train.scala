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
package com.intel.analytics.bigdl.example.chatbot

import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.models.rnn.Train._
import com.intel.analytics.bigdl.models.rnn.{SequencePreprocess, SimpleRNN, Utils}
import com.intel.analytics.bigdl.models.rnn.Utils.{TrainParams, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

class Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Train rnn on text")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val tokens = SequencePreprocess(
        param.dataFolder + "twitter_en.txt",
        sc = sc,
        param.sentFile,
        param.tokenFile)

      val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
        Array(param.trainingSplit, 1 - param.trainingSplit))

      val dictionary = Dictionary(tokens, param.vocabSize)
      val vocabSize = dictionary.getVocabSize()
      val padding = "###"
      dictionary.addWord(padding)
      dictionary.save(param.saveFolder)

      val maxTrainLength = tokens.map(x => x.length).max

      logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

      val padId = dictionary.getIndex(padding) + 1
      val padFeature = Tensor[Float](T(padding))
      val padLabel = Tensor[Float](T(padding))
      val featurePadding = PaddingParam(Some(Array(padFeature)),
        FixedLength(Array(maxTrainLength)))
      val labelPadding = PaddingParam(Some(Array(padLabel)),
        FixedLength(Array(maxTrainLength)))

      val trainSet = DataSet.rdd(tokens)
        .transform(TextToLabeledSentence[Float](dictionary))
        .transform(LabeledSentenceToSample[Float](totalVocabLength))
        .transform(SampleToMiniBatch[Float](
          param.batchSize,
          Some(featurePadding),
          Some(labelPadding)))

      val validationSet = DataSet.rdd(valtokens)
        .transform(TextToLabeledSentence[Float](dictionary, zeroIndexed = false))
        .transform(LabeledSentenceToSample[Float](oneHot = false))
        .transform(SampleToMiniBatch[Float](param.batchSize,
          Some(featurePadding), Some(labelPadding)))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val encoderCells =
          Array(
            Recurrent().add(LSTM(param.embedDim, param.embedDim)),
            Recurrent().add(LSTM(param.embedDim, param.embedDim)),
            Recurrent().add(LSTM(param.embedDim, param.embedDim))
          )

        val decoderCells =
          Array(
            Recurrent().add(LSTM(param.embedDim, param.embedDim)),
            Recurrent().add(LSTM(param.embedDim, param.embedDim)),
            Recurrent().add(LSTM(param.embedDim, param.embedDim))
          )

        var encoderRecs = Array(Recurrent().add(MultiRNNCell(decoderCells))
          .asInstanceOf[Recurrent[Double]])
        var decoderRecs = Array(RecurrentDecoder(seqLength).add(MultiRNNCell(decoderCells))
          .asInstanceOf[Recurrent[Double]])

        val model = Seq2seq(encoderRecs, decoderRecs,
          decoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)

        for (i <- 0 until 3) {
          model.forward(input).toTensor
          model.backward(input, gradOutput)
        }

        decoderRecs = decoderCells.map(Recurrent().add(_))
        val model2 = Seq2seq(encoderRecs, decoderRecs,
          decoderInputType = DecoderInputType.ENCODERINPUTSPLIT)
        curModel.reset()
        curModel
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = true)
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
          TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), sizeAverage = true))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
        .optimize()
      sc.stop()
    })
  }
}
