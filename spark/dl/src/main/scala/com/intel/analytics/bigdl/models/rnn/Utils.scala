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
import scopt.OptionParser
import java.io._

import com.intel.analytics.bigdl.dataset.text.LabeledSentence

import scala.util.Random
import org.apache.log4j.Logger

object Utils {
  case class TrainParams(
    folder: String =
      "./",
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    checkpoint: Option[String] = None,
    learningRate: Double = 0.1,
    momentum: Double = 0.0,
    weightDecay: Double = 0.0,
    dampening: Double = 0.0,
    hiddenSize: Int = 40,
    vocabSize: Int = 4000,
    bptt: Int = 4,
    nEpochs: Int = 30
  )

  val trainParser = new OptionParser[TrainParams]("BigDL SimpleRNN Train Example") {
    opt[String]('f', "folder")
      .text("where you put the text data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("checkpoint")
      .text("where to cache the model and state")
      .action((x, c) => c.copy(checkpoint = Some(x)))

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Double]('m', "momentum")
      .text("momentum")
      .action((x, c) => c.copy(momentum = x))

    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))

    opt[Double]("dampening")
      .text("dampening")
      .action((x, c) => c.copy(dampening = x))

    opt[Int]('h', "hidden")
      .text("hidden size")
      .action((x, c) => c.copy(hiddenSize = x))

    opt[Int]("vocab")
      .text("dictionary length | vocabulary size")
      .action((x, c) => c.copy(vocabSize = x))

    opt[Int]("bptt")
      .text("back propagation through time size")
      .action((x, c) => c.copy(bptt = x))

    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }

  case class TestParams(
     folder: String = "./",
     modelSnapshot: Option[String] = None,
     numOfWords: Option[Int] = None
  )

  val testParser = new OptionParser[TestParams]("BigDL rnn Test Example") {
    opt[String]('f', "folder")
      .text("where you put the text data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
      .required()

    opt[Int]("words")
      .text("number of words to write")
      .action((x, c) => c.copy(numOfWords = Some(x)))
      .required()
  }


  private[bigdl] def readSentence(directory: String)
  : Array[Array[String]] = {

    import scala.io.Source
    require(new File(directory + "/test.txt").exists(),
      s"test file ${directory + "/test.txt"} not exists!")
    val lines = Source.fromFile(directory + "/test.txt")
      .getLines().map(_.split("\\W+")).toArray
    lines
  }

  class Dictionary()
  extends Serializable {

    private var vocab2index: Map[String, Int] = null
    private var index2vocab: Map[Int, String] = null
    private var vocabLength: Int = 0
    private var discard: Array[String] = null
    private var discardLength: Int = 0

    val logger = Logger.getLogger(getClass)

    def this(directory: String) = {
      this()

      if (!new File(directory + "/dictionary.txt").exists()) {
        throw new IllegalArgumentException("dictionary file not exists!")
      }
      if (!new File(directory + "/discard.txt").exists()) {
        throw new IllegalArgumentException("discard file not exists!")
      }

      import scala.io.Source
      vocab2index = Source.fromFile(directory + "/dictionary.txt")
        .getLines.map(_.stripLineEnd.split("->", -1))
        .map(fields => fields(0).stripSuffix(" ") -> fields(1).stripPrefix(" ").toInt)
        .toMap[String, Int]
      index2vocab = vocab2index.map(x => (x._2, x._1))
      vocabLength = vocab2index.size
      discard = Source.fromFile(directory + "/discard.txt")
        .getLines().toArray
      discardLength = discard.length
    }

    def getIndex(word: String): Int = {
      vocab2index.getOrElse(word, vocabLength)
    }

    def getWord(index: Float): String = {
      getWord(index.toInt)
    }

    def getWord(index: Double): String = {
      getWord(index.toInt)
    }

    def getWord(index: Int): String = {
      index2vocab.getOrElse(index,
        discard(Random.nextInt(discardLength)))
    }

    def length(): Int = vocabLength

    def print(): Unit = {
      vocab2index.foreach(x =>
        logger.info(x._1 + " -> " + x._2))
    }

    def printDiscard(): Unit = {
      discard.foreach(x =>
        logger.info(x))
    }
  }

  class WordTokenizer(
    inputFile: String,
    saveDirectory: String,
    dictionaryLength: Int)
    extends Serializable {

    def process() {
      if (!new File(saveDirectory + "/mapped_data.txt").exists) {
        import scala.io.Source

        val lines = Source.fromFile(inputFile).getLines.toArray
          .filter(_.length > 0)

        // Special Words
        val sentence_start_token = "SENTENCE_START"
        val sentence_end_token = "SENTENCE_END"
        val unknown_token = "UNKNOWN_TOKEN"

        // Create dictionary with frequency as value for each word
        val sentences = lines.map(x => sentence_start_token + " " + x + " " + sentence_end_token)
        val freqDict = sentences.flatMap(_.split("\\W+"))
          .foldLeft(Map.empty[String, Int]) {
            (count, word) => count + (word -> (count.getOrElse(word, 0) + 1))
          }.toSeq.sortBy(_._2)

        // Select most common words
        val length = math.min(dictionaryLength - 1, freqDict.length)
        val vocabDict = freqDict.drop(freqDict.length - length).map(_._1)
        val vocabSize = vocabDict.length
        val word2index = vocabDict.zipWithIndex.toMap
        val discardDict = freqDict.take(freqDict.length - length).map(_._1)

        // save dictionary
        new PrintWriter(saveDirectory + "/dictionary.txt") {
          write(word2index.mkString("\n")); close
        }

        // save discard dictionary
        new PrintWriter(saveDirectory + "/discard.txt") {
          write(discardDict.mkString("\n")); close
        }

        // Convert the string texts to integer arrays
        val mappedDF = sentences.map(x => x.split("\\W+")
          .map(word => word2index.getOrElse(word, vocabSize)))

        // save converted data
        new PrintWriter(saveDirectory + "/mapped_data.txt") {
          write(mappedDF.map(_.mkString(",")).mkString("\n")); close
        }
      }
    }
  }

  private[bigdl] def loadInData(filedirect: String, dictionarySize: Int)
  : (Array[LabeledSentence[Float]], Array[LabeledSentence[Float]],
    Int, Int) = {

    import scala.io.Source

    val logData = Source.fromFile(filedirect + "/mapped_data.txt").getLines().toArray
    val dataFlow = logData.map(x => {
      val seq = x.split(",").toList.asInstanceOf[Seq[Int]]
      (seq.take(seq.length - 1), seq.drop(1))
    })

    val length = dataFlow.length
    val seq = Random.shuffle((1 to length).toList)
    val seqTrain = seq.take(Math.floor(seq.length*0.8).toInt).toArray
    val seqVal = seq.drop(Math.floor(seq.length*0.8).toInt).toArray

    val trainFlow = seqTrain.collect(dataFlow)
    val valFlow = seqVal.collect(dataFlow)

    var trainMaxLength = 0

    val trainData = trainFlow.map(x => {
      val data = x._1
      val label = x._2
      val numOfWords = data.length
      trainMaxLength = math.max(trainMaxLength, numOfWords)
      val input = new Array[Float](numOfWords)
      val target = new Array[Float](numOfWords)
      var i = 0
      while (i < numOfWords) {
        input(i) = data(i).toString.toInt.toFloat
        target(i) = label(i).toString.toInt.toFloat
        i += 1
      }
      new LabeledSentence[Float](input, target)
    })

    var valMaxLength = 0

    val valData = valFlow.map(x => {
      val data = x._1
      val label = x._2
      val numOfWords = data.length
      valMaxLength = math.max(valMaxLength, numOfWords)
      val input = new Array[Float](numOfWords)
      val target = new Array[Float](numOfWords)
      var i = 0
      while (i < numOfWords) {
        input(i) = data(i).toString.toInt.toFloat
        target(i) = label(i).toString.toInt.toFloat
        i += 1
      }
      new LabeledSentence[Float](input, target)
    })

    (trainData, valData, trainMaxLength, valMaxLength)
  }
}
