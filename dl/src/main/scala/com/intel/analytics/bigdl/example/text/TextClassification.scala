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

package com.intel.analytics.bigdl.example.text.classification

import java.io.File
import java.util
import java.util.{HashMap, Map => JMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

class TokensToBatchTransformer(batchSize: Int, tokenNum: Int, vectorDim: Int)
  extends Transformer[(Array[Array[Float]], Float), MiniBatch[Float]] {

  val log: Logger = LoggerFactory.getLogger(this.getClass)

  override def apply(prev: Iterator[(Array[Array[Float]], Float)]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        var i = 0
        while (i < batchSize && prev.hasNext) {
          val sample = prev.next()
          if (featureData == null) {
            featureData = new Array[Float](batchSize * tokenNum * vectorDim)
            labelData = new Array[Float](batchSize)
          }
          Array.copy(sample._1.flatten, 0,
            featureData, i * tokenNum * vectorDim, tokenNum * vectorDim)
          labelData(i) = sample._2
          i += 1
        }

        featureTensor.set(Storage[Float](featureData),
          storageOffset = 1, sizes = Array(i, tokenNum, vectorDim))
        labelTensor.set(Storage[Float](labelData),
          storageOffset = 1, sizes = Array(i))
        MiniBatch(featureTensor.transpose(2, 3), labelTensor)
      }
    }
  }
}

case class TextClassificationParams(baseDir: String = "./",
                                    maxSequenceLength: Int = 1000,
                                    maxWordsNum: Int = 20000,
                                    trainingSplit: Double = 0.8,
                                    batchSize: Int = 128,
                                    embeddingDim: Int = 100)

class TextClassification(param: TextClassificationParams) {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val gloveDir = s"${param.baseDir}/glove.6B/"
  val textDataDir = s"${param.baseDir}/20_newsgroup/"

  private def buildWord2Vec(): JMap[String, Array[Float]] = {
    val preWord2Vec = new HashMap[String, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename).getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      preWord2Vec.put(word, coefs)
    }
    log.info(s"Found ${preWord2Vec.size()} word vectors.")
    preWord2Vec
  }

  private def loadRawData(): (ArrayBuffer[ArrayBuffer[String]], ArrayBuffer[Int], Int) = {
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Int]()
    // category is a string name, label is it's index
    val categoryToLabel = new util.HashMap[String, Int]()
    val categoryPathList = new File(textDataDir).listFiles().filter(_.isDirectory).toList.sorted

    categoryPathList.foreach { categoryPath =>
      val label_id = categoryToLabel.size() + 1 // one-base index
      categoryToLabel.put(categoryPath.getName(), label_id)
      val textFiles = categoryPath.listFiles()
        .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
      textFiles.foreach { file =>
        val source = Source.fromFile(file, "ISO-8859-1")
        val text = try source.getLines().toList.mkString("\n") finally source.close()
        texts.append(text)
        labels.append(label_id)
      }
    }
    log.info(s"Found ${texts.length} texts.")
    val tokenizer = new SimpleTokenizer(nbWords = param.maxWordsNum)
    tokenizer.init(texts)
    val sequences = tokenizer.textsToSequences(texts, seqLen = param.maxSequenceLength)
    val wordSeqs = sequences.map {
      _.map {
        tokenizer.indexToWord(_)
      }
    }
    (wordSeqs, labels, categoryToLabel.size())
  }

  // Split the data into a training set and a validation set
  private def splitData(data: Array[Array[Array[Float]]], labels: Array[Float]) = {
    val totalSamples = data.length
    val shuffledIndexes = Random.shuffle(Range(0, data.length).toList)
    val trainingNum = (param.trainingSplit * totalSamples).toInt

    val xTrain = new Array[Array[Array[Float]]](trainingNum)
    val yTrain = new Array[Float](trainingNum)

    val xVal = new Array[Array[Array[Float]]](totalSamples - trainingNum)
    val yVal = new Array[Float](totalSamples - trainingNum)

    var i = 0
    while (i < totalSamples) {
      if (i < trainingNum) {
        xTrain(i) = data(shuffledIndexes(i))
        yTrain(i) = labels(shuffledIndexes(i))
      } else {
        xVal(i - trainingNum) = data(shuffledIndexes(i))
        yVal(i - trainingNum) = labels(shuffledIndexes(i))
      }
      i += 1
    }
    log.info(s"Size of training set: $trainingNum")
    log.info(s"Size of validation set: ${totalSamples - trainingNum}")
    (xTrain, yTrain, xVal, yVal)
  }

  // TODO: Move this to the embedding layer.
  private def embedding(sequences: ArrayBuffer[ArrayBuffer[String]]): Array[Array[Array[Float]]] = {
    log.info("Indexing word vectors.")
    val word2Vec = buildWord2Vec()
    val data = sequences.map {
      _.map { word =>
        if (word2Vec.containsKey(word)) {
          word2Vec.get(word)
        } else {
          // Treat it as zeros if cannot be found from pre-trained word2Vec
          Array.fill[Float](param.embeddingDim)(0)
        }
      }.toArray
    }.toArray
    data
  }

  private def toDataSet(data: Array[Array[Array[Float]]], labels: Array[Float])
    : DataSet[MiniBatch[Float]] = {
    val result = DataSet.array(data.zip(labels)).transform(new TokensToBatchTransformer(
      param.batchSize, param.maxSequenceLength, param.embeddingDim
    ))
    result
  }

  def train(): Unit = {
    val (sequences, labels, classNum) = loadRawData()
    val data = embedding(sequences)
    val (xTrain, yTrain, xVal, yVal) = splitData(data, labels.map(_.toFloat).toArray)

    // TODO: Replace SpatialConv and SpatialMaxPolling with 1D implementation
    val model = Sequential[Float]()

    model.add(Reshape(Array(param.embeddingDim, 1, param.maxSequenceLength)))

    model.add(SpatialConvolution(param.embeddingDim, 128, 5, 1, initMethod = Xavier))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1, initMethod = Xavier))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1, initMethod = Xavier))
    model.add(ReLU())

    model.add(SpatialMaxPooling(35, 1, 35, 1))

    model.add(Reshape(Array(128)))
    model.add(Linear(128, 100))
    model.add(Linear(100, classNum))
    model.add(LogSoftMax())

    // Start to train and SGD is the default optimizer
    val optimizer = Optimizer(
      model = model,
      dataset = toDataSet(xTrain, yTrain),
      criterion = new ClassNLLCriterion[Float]()
    )

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)
    optimizer
      .setState(state)
      .setOptimMethod(new Adagrad())
      .setValidation(Trigger.everyEpoch, toDataSet(xVal, yVal), Array(new Top1Accuracy[Float]))
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()
  }
}

object TextClassification {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  def main(args: Array[String]): Unit = {
    val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
      opt[String]('b', "baseDir")
        .required()
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]('s', "maxSequenceLength")
        .text("maxSequenceLength")
        .action((x, c) => c.copy(maxSequenceLength = x.toInt))
      opt[String]('w', "maxWordsNum")
        .text("maxWordsNum")
        .action((x, c) => c.copy(maxWordsNum = x.toInt))
      opt[String]('l', "trainingSplit")
        .text("trainingSplit")
        .action((x, c) => c.copy(trainingSplit = x.toDouble))
      opt[String]('z', "batchSize")
        .text("batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[String]('e', "embeddingDim")
        .text("embeddingDim")
        .action((x, c) => c.copy(embeddingDim = x.toInt))
    }

    localParser.parse(args, TextClassificationParams()).map { param =>
      log.info(s"Current parameters: $param")
      val textClassification = new TextClassification(param)
      textClassification.train()
    }
  }
}
