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

import com.intel.analytics.bigdl.dataset.text.LabeledSentence

import scala.util.Random

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
    nEpochs: Int = 30,
    coreNumber: Int = -1)

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

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
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

        // save dictionary
        new PrintWriter(saveDirectory + "/dictionary.txt") {
          write(word2index.mkString("\n")); close
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

  private[bigdl] def readSentence(filedirect: String, dictionarySize: Int)
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
