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

import java.util.{HashMap => JHashMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.dataset.text.{SentenceTokenizer, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.JTensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, File, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

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
        .set("spark.driver.maxResultSize", "3g")
      val sc = new SparkContext(conf)
      Engine.init

      val text = false
      val (trainSet, validationSet, vocabSize, dictionary, padId) = if (text) {
        val lines = Source
          .fromFile(param.dataFolder + "twitter_en.txt", "UTF-8")
          .getLines
          .toList
          .zipWithIndex

        val evenLines = lines.filter(x => x._2 % 2 == 0)
          .map(_._1).toIterator
        val oddLines = lines.filter(x => x._2 % 2 != 0)
          .map(_._1).toIterator

        val evenTokens = SentenceTokenizer().apply(evenLines)
        val oddTokens = (SentenceBiPadding() -> SentenceTokenizer()).apply(oddLines)

        val sample = evenTokens.zip(oddTokens).toSeq
        val tokens = sc.parallelize(sample)

        val dictionary = Dictionary(tokens.map(_._1) ++ tokens.map(_._2), param.vocabSize)
        val padding = "###"
        dictionary.addWord(padding)
        dictionary.save(param.saveFolder)
        val vocabSize = dictionary.getVocabSize() + 1

        val padId = dictionary.getIndex(padding) + 1

        val Array(trainRDD, valRDD) = tokens.randomSplit(
          Array(param.trainingSplit, 1 - param.trainingSplit))

        val trainSet = trainRDD
          .map(chatToLabeledChat(dictionary, _))
          .map(x => labeledChatToSample(x))

        val validationSet = valRDD
          .map(chatToLabeledChat(dictionary, _))
          .map(x => labeledChatToSample(x))
        (trainSet, validationSet, vocabSize, dictionary, padId)
      } else {
        val idx2w = Source
          .fromFile(param.dataFolder + "idx2w.csv", "UTF-8")
          .getLines
          .map(x => {
            val split = x.split(",")
            (split(0).toInt, if (split.length < 2) "" else split(1))
          })
          .toMap

        val w2idx = Source
          .fromFile(param.dataFolder + "w2idx.csv", "UTF-8")
          .getLines
          .map(x => {
            val split = x.split(",")
            (split(0), split(1).toInt)
          })
          .toMap

        val dictionary = Dictionary(idx2w, w2idx)
        val vocabSize = dictionary.getVocabSize()
        val padId = dictionary.getIndex("_") + 1
        val chat1 = Source
          .fromFile(param.dataFolder + "chat1_1.txt", "UTF-8")
          .getLines
          .toList
          .map(_.split(",").map(_.toInt))
          .map(s => s.filter(id => id != 0))
        val chat2List = Source
          .fromFile(param.dataFolder + "chat2_1.txt", "UTF-8")
          .getLines
          .toList
          .toIterator

        val chat2 = SentenceIdxBiPadding(dictionary = dictionary)
          .apply(chat2List)
          .map(_.split(",").map(_.toInt))
          .map(s => s.filter(id => id != 0))
          .toList

        val tokens = sc.parallelize(chat1.zip(chat2))

        val trainRDD = tokens

        val trainSet = trainRDD
          .map(chatIdxToLabeledChat(_))
          .map(labeledChatToSample(_))

        val validationSet = trainSet
        (trainSet, validationSet, vocabSize, dictionary, padId)
      }
      val padFeature = Tensor[Float](T(padId))
      val padLabel = Tensor[Float](T(padId))



        val encoder =
          Array(
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim)),
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim)),
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim))
          )

        val decoder =
          Array(
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim)),
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim)),
            Recurrent(maskZero = true).add(LSTM(param.embedDim, param.embedDim))
          )

        val enclookuptable = LookupTable(
          vocabSize,
          param.embedDim,
          paddingValue = padId,
          maskZero = true
        )

        val declookuptable = LookupTable(
          vocabSize,
          param.embedDim,
          paddingValue = padId,
          maskZero = true
        )

      declookuptable.weight = enclookuptable.weight
      declookuptable.gradWeight = enclookuptable.gradWeight

      val preEncoder = enclookuptable
      val preDecoder = declookuptable

        var model: Module[Float] = Sequential()
          .add(
            Seq2seq(encoder, decoder,
              preEncoder = preEncoder,
              preDecoder = preDecoder,
              decoderInputType = DecoderInputType.ENCODERINPUTSPLIT))
          .add(TimeDistributed(Linear(param.embedDim, vocabSize), maskZero = true))
          .add(TimeDistributed(LogSoftMax()))

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adam[Float](learningRate = 0.0001)
      }

      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainSet,
        criterion = TimeDistributedMaskCriterion(
          ClassNLLCriterion(paddingValue = padId, sizeAverage = false),
          paddingValue = padId
        ),
        batchSize = param.batchSize,
        featurePaddingParam = PaddingParam[Float](
          paddingTensor =
            Some(Array(padFeature, padFeature))),
        labelPaddingParam = PaddingParam[Float](
          paddingTensor =
            Some(Array(padLabel))))

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.severalIteration(20))
      }

      if (param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setOptimMethod(optimMethod)
        .setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)

      val seeds = Array("happy birthday have a nice day",
        "donald trump won last nights presidential debate according to snap online polls")


      var i = 1
      while (i <= param.nEpochs) {
        val p = model.getParameters()
        optimizer
          .setEndWhen(Trigger.maxEpoch(i))
        model = optimizer.optimize()

        model.evaluate()
        for (seed <- seeds) {
          println("Query> " + seed)
          val evenToken = SentenceTokenizer().apply(Array(seed).toIterator).toArray
          val oddToken = (SentenceBiPadding() -> SentenceTokenizer())
            .apply(Array("").toIterator).toArray
          val labeledChat = evenToken.zip(oddToken)
            .map(chatToLabeledChat(dictionary, _)).apply(0)

          val sent1 = Tensor(Storage(labeledChat._1), 1, Array(1, labeledChat._1.length))
          var sent2 = Tensor(Storage(labeledChat._2), 1, Array(1, labeledChat._2.length))
          val timeDim = 2
          val featDim = 3
          val concat = Tensor[Float]()
          var curInput = sent2
          val end = dictionary.getIndex(SentenceToken.end) + 1
          var break = false

          var i = 0
          // Iteratively output predicted words
          while (i < 30 && !break) {
            val output = model.forward(T(sent1, curInput)).toTensor[Float]
            val predict = output.max(featDim)._2
              .select(timeDim, output.size(timeDim)).valueAt(1, 1).toInt
            if (predict == end) break = true
            if (!break) {
              concat.resize(1, curInput.size(timeDim) + 1)
              concat.narrow(timeDim, 1, curInput.size(timeDim)).copy(curInput)
              concat.setValue(1, concat.size(timeDim), predict)
              curInput.resizeAs(concat).copy(concat)
            }
            i += 1
          }
          val predArray = new Array[Float](curInput.nElement())
          Array.copy(curInput.storage().array(), curInput.storageOffset() - 1,
            predArray, 0, curInput.nElement())
          val result = predArray.grouped(curInput.size(timeDim)).toArray[Array[Float]]
            .map(x => x.map(t => dictionary.getWord(t - 1)))
          println(result.map(x => x.mkString(" ")).mkString("\n"))
        }
        model.clearState()
        i += 1
      }
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

  def chatIdxToLabeledChat[T: ClassTag](
    chat: (Array[Int], Array[Int]))(implicit ev: TensorNumeric[T])
  : (Array[T], Array[T], Array[T]) = {
    val (indices1, indices2) =
      (chat._1.map(x => ev.fromType[Int](x + 1)),
        chat._2.map(x => ev.fromType[Int](x + 1)))
    val label = indices2.drop(1)
    (indices1, indices2.take(indices2.length - 1), label)
  }

  class SentenceIdxBiPadding(
    start: Option[String] = None,
    end: Option[String] = None,
    dictionary: Dictionary
  )
    extends Transformer[String, String] {

    val sentenceStart = dictionary.getIndex(start.getOrElse(SentenceToken.start))
    val sentenceEnd = dictionary.getIndex(end.getOrElse(SentenceToken.end))

    override def apply(prev: Iterator[String]): Iterator[String] = {
      prev.map(x => {
        val sentence = sentenceStart + "," + x + "," + sentenceEnd
        sentence
      })
    }
  }

  object SentenceIdxBiPadding {
    def apply(
      start: Option[String] = None,
      end: Option[String] = None,
      dictionary: Dictionary
    ): SentenceIdxBiPadding = new SentenceIdxBiPadding(start, end, dictionary)
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
