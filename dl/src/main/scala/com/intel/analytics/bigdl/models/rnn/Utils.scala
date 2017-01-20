/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
import scopt.OptionParser
import java.io._

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.{Dictionary, DocumentTokenizer, LabeledSentence}

import scala.util.Random
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer

object Utils {
  val logger = Logger.getLogger(getClass)
  case class TrainParams(
    dataFolder: String = "./",
    saveFolder: String = "./",
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
    nEpochs: Int = 30,
    coreNumber: Int = -1)

  val trainParser = new OptionParser[TrainParams]("BigDL SimpleRNN Train Example") {
    opt[String]('f', "dataFolder")
      .text("where you put the text data")
      .action((x, c) => c.copy(dataFolder = x))
      .required()

    opt[String]('s', "saveFolder")
      .text("where you save the processed text data")
      .action((x, c) => c.copy(saveFolder = x))
      .required()

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

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
  }

  case class TestParams(
     folder: String = "./",
     test: String = "test.txt",
     modelSnapshot: Option[String] = None,
     stateSnapshot: Option[String] = None,
     numOfWords: Option[Int] = None,
     coreNumber: Int = -1)

  val testParser = new OptionParser[TestParams]("BigDL rnn Test Example") {
    opt[String]('f', "folder")
      .text("where to load and write data")
      .action((x, c) => c.copy(folder = x))
      .required()

    opt[String]('t', "test")
      .text("test file name, should be under the folder")
      .action((x, c) => c.copy(test = x))
      .required()

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
      .required()

    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
      .required()

    opt[Int]("words")
      .text("number of words to write")
      .action((x, c) => c.copy(numOfWords = Some(x)))
      .required()

    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
  }


  private[bigdl] def readSentence(directory: String,
                                  test: String)
  : Array[Array[String]] = {

    val testFile = new File(directory, test)
    if (!testFile.exists()) {
      throw new IllegalArgumentException("test file not exists!")
    }

    val dataSet = DataSet.array(Array(testFile.getAbsolutePath))
    val documentTokenizer = DocumentTokenizer()
    val writePipeLine = dataSet -> documentTokenizer
    val iter = writePipeLine.toLocal().data(train = false)

    iter.next.map(x =>
      Array(Sentence.sentenceStart) ++ x.take(x.length - 1))
  }

  class WordTokenizer(
    dataFolder: String,
    saveFolder: String,
    dictionaryLength: Int)
    extends Serializable {

    private var _vocabSize: Int = 0

    def length(): Int = _vocabSize

    def process() {
      val inputDirect = new File(dataFolder)
      val saveTo = new File(saveFolder)

      if (!inputDirect.isDirectory || inputDirect.list.length == 0) {
        throw new IllegalArgumentException(
          "dataFolder is invalid or input files not exists!")
      } else if (!saveTo.isDirectory) {
        throw new IllegalArgumentException("saveFolder is invalid!")
      } else if (!saveTo.list.contains("mapped_data.txt")) {

        val dataSet = DataSet.array(inputDirect.list.map(x =>
        inputDirect.getAbsolutePath + "/" + x))
        val documentTokenizer = DocumentTokenizer()
        val writePipeLine = dataSet -> documentTokenizer
        val iter = writePipeLine.toLocal().data(train = false)

        val lines = ArrayBuffer[Array[String]]()
        while (iter.hasNext) {
          iter.next().foreach(x => {
            lines.append(
              Array(Sentence.sentenceStart) ++ x ++ Array(Sentence.sentenceEnd))
          })
        }
        val sentences = lines.toArray

        val dictionary = Dictionary(sentences, dictionaryLength)
        _vocabSize = dictionary.vocabSize() + 1

        // save dictionary
        new PrintWriter(saveTo.getAbsolutePath + "/dictionary.txt") {
          write(dictionary.word2Index().mkString("\n")); close
        }

        // save discard dictionary
        new PrintWriter(saveTo.getAbsolutePath + "/discard.txt") {
          write(dictionary.discardVocab().mkString("\n")); close
        }

        // Convert the string texts to integer arrays
        val mappedDF = sentences.filter(_.size > 3)
          .map(word => word.map(
            w => dictionary.word2Index().getOrElse(w, dictionary.vocabSize())))

        // save converted data
        new PrintWriter(saveTo.getAbsolutePath + "/mapped_data.txt") {
          write(mappedDF.map(_.mkString(",")).mkString("\n")); close
        }
      } else {
        logger.info("mapped_data.txt already exists! Dictionary Size unchanged.")
        val dictionary = Dictionary(saveTo.getAbsolutePath)
        _vocabSize = dictionary.vocabSize() + 1
      }
    }
  }

  private[bigdl] def loadData(filedirect: String, dictionarySize: Int)
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

object Sentence {
  val sentenceStart = "SENTENCE_START"
  val sentenceEnd = "SENTENCE_END"
}
