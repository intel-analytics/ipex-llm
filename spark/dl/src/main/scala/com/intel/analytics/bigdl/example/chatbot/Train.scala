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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.text.{SentenceTokenizer, _}
import com.intel.analytics.bigdl.models.rnn.SimpleRNN
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature
import org.apache.spark.rdd.RDD

import scala.collection.Iterator
import scala.io.Source
import scala.reflect.ClassTag

object Train {
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

      val lines = Source
        .fromFile(param.dataFolder + "twitter_en.txt")
        .getLines
        .toList
        .zipWithIndex

      val evenLines = lines.filter(x => x._2 % 2 == 0)
        .map(_._1).toIterator
      val oddLines = lines.filter(x => x._2 % 2 != 0)
        .map(_._1).toIterator

      val evenTokens = (SentenceBiPadding() -> SentenceTokenizer()).apply(evenLines)
      val oddTokens = (SentenceBiPadding() -> SentenceTokenizer()).apply(oddLines)

      val sample = evenTokens.zip(oddTokens).toSeq
      val tokens = sc.parallelize(sample)

      val Array(trainRDD, valRDD) = tokens.randomSplit(
        Array(param.trainingSplit, 1 - param.trainingSplit))

      val dictionary = Dictionary(tokens.map(_._1) ++ tokens.map(_._2), param.vocabSize)
      val padding = "###"
      dictionary.addWord(padding)
      dictionary.save(param.saveFolder)
      val vocabSize = dictionary.getVocabSize() + 1

      val maxTrainLength = tokens.map(x => math.max(x._1.length, x._2.length)).max

      //      logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

      val padId = dictionary.getIndex(padding) + 1
      val padFeature = Tensor[Float](T(padId))
      val padLabel = Tensor[Float](T(padId))

      val trainSet = trainRDD
        .map(chatToLabeledChat(dictionary, _))
        .map(x => labeledChatToSample(x))

      val validationSet = valRDD
        .map(chatToLabeledChat(dictionary, _))
        .map(x => labeledChatToSample(x))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val encoder =
          Array(
//            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh())),
//            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh())),
            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh()))
          )

        val decoder =
          Array(
//            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh())),
//            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh())),
            Recurrent().add(RnnCell(param.embedDim, param.embedDim, Tanh()))
          )
        val enclookuptable = LookupTable(vocabSize, param.embedDim)
        val (enclookuptableW, enclookuptableG) = enclookuptable.getParameters()

        val declookuptable = LookupTable(vocabSize, param.embedDim)
        val preEncoder = enclookuptable
        val preDecoder = declookuptable

        val model = Sequential()
          .add(
            Seq2seq(encoder, decoder,
              preEncoder = preEncoder,
              preDecoder = preDecoder,
//              preEncoder = LookupTable(vocabSize, param.embedDim),
//              preDecoder = LookupTable(vocabSize, param.embedDim),
              decoderInputType = DecoderInputType.ENCODERINPUTSPLIT))
          .add(TimeDistributed(Linear(param.embedDim, vocabSize)))
          .add(TimeDistributed(LogSoftMax()))
        model.reset()
        model
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening)
      }

      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainSet,
        criterion = TimeDistributedCriterion(ClassNLLCriterion()),
        batchSize = param.batchSize,
        featurePaddingParam = PaddingParam[Float](
          paddingTensor =
            Some(Array(padFeature, padFeature))),
        labelPaddingParam = PaddingParam[Float](
          paddingTensor =
            Some(Array(padLabel))))

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      if (param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setValidation(
          Trigger.everyEpoch,
          validationSet,
          Array(new Loss[Float](
            TimeDistributedCriterion(ClassNLLCriterion()))),
          param.batchSize,
          featurePaddingParam = PaddingParam[Float](
            paddingTensor =
              Some(Array(padFeature, padFeature))),
          labelPaddingParam = PaddingParam[Float](
            paddingTensor =
              Some(Array(padLabel))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
        .optimize()
      sc.stop()
    })
  }


  def chatToLabeledChat[T: ClassTag](
    dictionary: Dictionary,
    chat: (Array[String], Array[String]))(implicit ev: TensorNumeric[T])
  : (Array[T], Array[T], Array[T]) = {
    val (indices1, indices2) =
      (chat._1.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)),
        chat._2.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)))
    val label = indices2.drop(1)
    (indices1, indices2.take(indices2.length - 1), label)
  }

  def labeledChatToSample[T: ClassTag](
    labeledChat: (Array[T], Array[T], Array[T]))
    (implicit ev: TensorNumeric[T]): Sample[T] = {

    val sentence1: Tensor[T] = Tensor(Storage(labeledChat._1))
    val sentence2: Tensor[T] = Tensor(Storage(labeledChat._2))
    val label: Tensor[T] = Tensor(Storage(labeledChat._3))

    Sample(featureTensors = Array(sentence1, sentence2), labelTensor = label)
  }
}
