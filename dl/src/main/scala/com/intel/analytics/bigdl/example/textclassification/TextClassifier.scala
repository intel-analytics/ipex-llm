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

package com.intel.analytics.bigdl.example.textclassification

import java.io.File
import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source
import scala.util.Random

/**
 * This example use a (pre-trained GloVe embedding) to convert word to vector,
 * and uses it to train a text classification model on the 20 Newsgroup dataset
 * with 20 different categories. This model can achieve around 90% accuracy after
 * 2 epoches training.
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
  private def buildWord2Vec(): Map[String, Array[Float]] = {
    val preWord2Vec = MMap[String, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.100d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      preWord2Vec.put(word, coefs)
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

  private def transforming(dataSet: DataSet[(String, Float)],
                        word2Meta: Map[String, WordMeta],
                        word2Vec: Map[String, Array[Float]])
    : DataSet[MiniBatch[Float]] = {
  // You can implement this via pure RDD operation if on Spark mode only.
    val result = dataSet -> Tokens(word2Meta) -> Shapping(param.maxSequenceLength) ->
      Vectorization(param.embeddingDim, word2Vec) -> Batching(
      param.batchSize, Array(param.maxSequenceLength, param.embeddingDim))
    result
  }

  /**
   * Go through the whole data set to gather some meta info for the tokens.
   * Tokens would be discarded if the frequency ranking is less then maxWordsNum
   */
  def analyzeTexts(dataRdd: RDD[(String, Float)]): Map[String, WordMeta] = {
    val frequencies = dataRdd.flatMap{case (text: String, label: Float) =>
      Tokens.toTokens(text)
    }.map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(0, param.maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map{item =>
      (item._1._1, WordMeta(item._1._2, item._2))}.toMap
    word2Meta
  }

  /**
   * Go through the whole data set to gather some meta info for the tokens.
   * Tokens would be discarded if the frequency ranking is less then maxWordsNum
   */
  def analyzeTexts(data: ArrayBuffer[(String, Float)]): Map[String, WordMeta] = {
    val frequencies = data.flatMap{case (text: String, label: Float) =>
      Tokens.toTokens(text)
    }.groupBy((word: String) => word)
     .mapValues(_.length).toList
     .sortBy(- _._2).slice(0, param.maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map{item =>
      (item._1._1, WordMeta(item._1._2, item._2))}.toMap
    word2Meta
  }

  // TODO: Replace SpatialConv and SpatialMaxPolling with 1D implementation
  def buildModel(classNum: Int): Sequential[Float] = {
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
    model
  }

  def train(): Unit = {
    val scOption = Engine.init(param.nodeNum, param.coreNum,
      param.env == "spark").map(conf => {
      conf.setAppName("Text classification")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      new SparkContext(conf)
    })

    log.info("Indexing word vectors.")
    val word2Vec = buildWord2Vec()
    val data = loadRawData()
    val (trainingDataSet, valDataSet) = scOption match {
      case Some(sc) =>
        // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
        val dataRdd = scOption.get.parallelize(data, param.partitionNum)
        val word2Meta = analyzeTexts(dataRdd)
        val Array(trainingRDD, valRDD) = dataRdd.randomSplit(
          Array(param.trainingSplit, 1 - param.trainingSplit))
        (transforming(DataSet.rdd(trainingRDD, param.partitionNum), word2Meta, word2Vec),
          transforming(DataSet.rdd(trainingRDD, param.partitionNum), word2Meta, word2Vec))
      case _ =>
        val word2Meta = analyzeTexts(data)
        val (trainingData, valData) =
          Random.shuffle(data).splitAt((data.length * param.trainingSplit).toInt)
        (transforming(DataSet.array(trainingData.toArray), word2Meta, word2Vec),
          transforming(DataSet.array(valData.toArray), word2Meta, word2Vec))
    }

    val optimizer = Optimizer(
      model = buildModel(classNum),
      dataset = trainingDataSet,
      criterion = new ClassNLLCriterion[Float]()
    )

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)
    optimizer
      .setState(state)
      .setOptimMethod(new Adagrad())
      .setValidation(Trigger.everyEpoch,
        valDataSet, Array(new Top1Accuracy[Float]))
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()
    scOption.map(_.stop())
  }
}

/**
 * @param baseDir The root directory which containing the training and embedding data
 * @param maxSequenceLength number of the tokens
 * @param maxWordsNum maximum word to be included
 * @param trainingSplit percentage of the training data
 * @param batchSize size of the mini-batch
 * @param embeddingDim size of the embedding vector
 * @param coreNum same idea of spark core
 * @param nodeNum size of the cluster
 * @param partitionNum partition number of a training RDD.
 *   It should be equal to nodeNum(We might relax this in the following release)
 * @param env spark mode or no-spark mode
 */
case class TextClassificationParams(baseDir: String = "./",
  maxSequenceLength: Int = 1000,
  maxWordsNum: Int = 20000,
  trainingSplit: Double = 0.8,
  batchSize: Int = 128,
  embeddingDim: Int = 100,
  coreNum: Int = 1,
  nodeNum: Int = 1,
  partitionNum: Int = 1,
  env: String = "nospark")

object TextClassifier {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  def main(args: Array[String]): Unit = {
    val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
      opt[String]('b', "baseDir")
        .required()
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]('e', "env")
        .required()
        .text("spark or without spark, possible input: spark, nospark")
        .action((x, c) => c.copy(env = x))
      opt[String]('p', "partitionNum")
        .text("you may want to tune the partitionNum if run into spark mode")
        .action((x, c) => c.copy(partitionNum = x.toInt))

      opt[String]('o', "coreNum")
        .text("core number")
        .action((x, c) => c.copy(coreNum = x.toInt))
      opt[String]('n', "nodeNum")
        .text("nodeNumber")
        .action((x, c) => c.copy(nodeNum = x.toInt))
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
