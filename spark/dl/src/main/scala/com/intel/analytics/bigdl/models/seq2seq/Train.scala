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
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentenceToSample, TextToLabeledSentence, TextToLabeledSentence2}
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
import com.intel.analytics.bigdl.models.rnn.Utils.TrainParams
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Train2 {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    import utils._
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf()
        .setAppName("Train Seq2seq on fake data")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening)
      }

      val tokens = SequencePreprocess(
        param.dataFolder + "/train.txt",
        sc = sc,
        param.sentFile,
        param.tokenFile)

      val dictionary = Dictionary(tokens, param.vocabSize)
      dictionary.save(param.saveFolder)

      val maxTrainLength = tokens.map(x => x.length).max
      //      val tokensNew = tokens.filter(x => x.length <= maxTrainLength)
      val tokensNew = tokens
      println(s"maxTrain length = ${maxTrainLength}")

      val totalVocabLength = dictionary.getVocabSize() + 1
      val startIdx = dictionary.getIndex(SentenceToken.start)
      val endIdx = dictionary.getIndex(SentenceToken.end)
      val padFeature = Tensor[Float]().resize(totalVocabLength)
      padFeature.setValue(endIdx + 1, 1.0f)
      val padLabel = Tensor[Float](T(startIdx.toFloat + 1.0f))
      val featurePadding = PaddingParam(Some(Array(padFeature)),
        FixedLength(Array(maxTrainLength / 2)))
      val labelPadding = PaddingParam(Some(Array(padLabel)),
        FixedLength(Array(maxTrainLength / 2)))

      val trainSet = DataSet.rdd(tokensNew)
        .transform(TextToLabeledSentence2[Float](dictionary))
        .transform(LabeledSentenceToSample[Float](totalVocabLength))
        .transform(SampleToMiniBatch[Float](
          param.batchSize,
          Some(featurePadding),
          Some(labelPadding)))

      val inputSize = totalVocabLength
      val hiddenSize = param.hiddenSize

      val encoderRec = Array(Recurrent()
        .add(RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]())))

      val decoderRec = Array(RecurrentDecoder(maxTrainLength / 2)
        .add(RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]()))
        .asInstanceOf[Recurrent[Float]])

      //      val encoderRec = Array(Recurrent()
      //        .add(RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]())),
      //        Recurrent()
      //          .add(RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]())))
      //      val decoderCell = Array(RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]()),
      //        RnnCell[Float](hiddenSize, hiddenSize, Tanh[Float]()).asInstanceOf[Cell[Float]])
      //      val decoderRec = Array(RecurrentDecoder(maxTrainLength / 2).add(MultiRNNCell(decoderCell))
      //        .asInstanceOf[Recurrent[Float]])

      val model = Sequential()
        .add(TimeDistributed[Float](Linear[Float](inputSize, hiddenSize)))
      val seq2seq = Seq2seq(encoderRec, decoderRec)
      model.add(seq2seq)
      model.add(TimeDistributed[Float](Linear[Float](hiddenSize, totalVocabLength)))

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = true))

      optimizer
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()

      sc.stop()
    })
  }
}
