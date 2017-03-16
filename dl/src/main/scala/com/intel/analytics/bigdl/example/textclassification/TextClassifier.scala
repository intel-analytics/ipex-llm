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

package com.intel.analytics.bigdl.example.textclassification

import java.io.File
import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.example.textclassification.SimpleTokenizer._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source
import scala.language.existentials

  /**
 * This example use a (pre-trained GloVe embedding) to convert word to vector,
 * and uses it to train a text classification model on the 20 Newsgroup dataset
 * with 20 different categories. This model can achieve around 90% accuracy after
 * 2 epochs training.
 */
class TextClassifier(param: TextClassificationParams) {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val gloveDir = s"${param.baseDir}/glove.6B/"
  val textDataDir = s"${param.baseDir}/20_newsgroup/"
  var classNum = -1

  /**
   * Load the pre-trained word2Vec
   * @return A map from word to vector
   */
  private def buildWord2Vec(word2Meta: Map[String, WordMeta]): Map[Float, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

  /**
   * Load the training data from the given baseDir
   * @return An array of sample
   */
  private def loadRawData(): ArrayBuffer[(String, Float)] = {
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Float]()
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
    this.classNum = labels.toSet.size
    log.info(s"Found ${texts.length} texts.")
    log.info(s"Found ${classNum} classes")
    texts.zip(labels)
  }

  /**
   * Go through the whole data set to gather some meta info for the tokens.
   * Tokens would be discarded if the frequency ranking is less then maxWordsNum
   */
  def analyzeTexts(dataRdd: RDD[(String, Float)])
  : (Map[String, WordMeta], Map[Float, Array[Float]]) = {
    // Remove the top 10 words roughly, you might want to fine tuning this.
    val frequencies = dataRdd.flatMap{case (text: String, label: Float) =>
      SimpleTokenizer.toTokens(text)
    }.map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(10, param.maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map{item =>
      (item._1._1, WordMeta(item._1._2, item._2))}.toMap
    (word2Meta, buildWord2Vec(word2Meta))
  }

  // TODO: Replace SpatialConv and SpatialMaxPolling with 1D implementation
  def buildModel(classNum: Int): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Reshape(Array(param.embeddingDim, 1, param.maxSequenceLength)))

    model.add(SpatialConvolution(param.embeddingDim, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(5, 1, 5, 1))

    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())

    model.add(SpatialMaxPooling(35, 1, 35, 1))

    model.add(Reshape(Array(128)))
    model.add(Linear(128, 100))
    model.add(Linear(100, classNum))
    model.add(LogSoftMax())
    model
  }

  def train(): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Text classification")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init
    val sequenceLen = param.maxSequenceLength
    val embeddingDim = param.embeddingDim
    val trainingSplit = param.trainingSplit

    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
    val dataRdd = sc.parallelize(loadRawData(), param.partitionNum)
    val (word2Meta, word2Vec) = analyzeTexts(dataRdd)
    val word2MetaBC = sc.broadcast(word2Meta)
    val word2VecBC = sc.broadcast(word2Vec)
    val vectorizedRdd = dataRdd
        .map {case (text, label) => (toTokens(text, word2MetaBC.value), label)}
        .map {case (tokens, label) => (shaping(tokens, sequenceLen), label)}
        .map {case (tokens, label) => (vectorization(
          tokens, embeddingDim, word2VecBC.value), label)}
    val sampleRDD = vectorizedRdd.map {case (input: Array[Array[Float]], label: Float) =>
          Sample(
            featureTensor = Tensor(input.flatten, Array(sequenceLen, embeddingDim))
              .transpose(1, 2).contiguous(),
            labelTensor = Tensor(Array(label), Array(1)))
        }

    val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
      Array(trainingSplit, 1 - trainingSplit))

    val optimizer = Optimizer(
      model = buildModel(classNum),
      sampleRDD = trainingRDD,
      criterion = new ClassNLLCriterion[Float](),
      batchSize = param.batchSize
    )
    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)
    optimizer
      .setState(state)
      .setOptimMethod(new Adagrad())
      .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy[Float]), param.batchSize)
      .setEndWhen(Trigger.maxEpoch(20))
      .optimize()
    sc.stop()
  }
}

/**
 * @param baseDir The root directory which containing the training and embedding data
 * @param maxSequenceLength number of the tokens
 * @param maxWordsNum maximum word to be included
 * @param trainingSplit percentage of the training data
 * @param batchSize size of the mini-batch
 * @param embeddingDim size of the embedding vector
 */
case class TextClassificationParams(baseDir: String = "./",
  maxSequenceLength: Int = 1000,
  maxWordsNum: Int = 20000,
  trainingSplit: Double = 0.8,
  batchSize: Int = 128,
  embeddingDim: Int = 100,
  partitionNum: Int = 4)

object TextClassifier {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  Logger4j.getLogger("org").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("akka").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("breeze").setLevel(Levle4j.ERROR)
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def main(args: Array[String]): Unit = {
    val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
      opt[String]('b', "baseDir")
        .required()
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]('p', "partitionNum")
        .text("you may want to tune the partitionNum if run into spark mode")
        .action((x, c) => c.copy(partitionNum = x.toInt))
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
    }

    localParser.parse(args, TextClassificationParams()).map { param =>
      log.info(s"Current parameters: $param")
      val textClassification = new TextClassifier(param)
      textClassification.train()
    }
  }
}
