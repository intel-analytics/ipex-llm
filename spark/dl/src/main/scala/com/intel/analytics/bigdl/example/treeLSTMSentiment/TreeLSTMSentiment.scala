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

package com.intel.analytics.bigdl.example.treeLSTMSentiment

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.example.treeLSTMSentiment.TreeLSTMSentiment.TreeLSTMSentimentParam
import com.intel.analytics.bigdl.example.utils._
import com.intel.analytics.bigdl.example.utils.SimpleTokenizer._
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
// import com.intel.analytics.bigdl.example.utils._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.LoggerFilter
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser
import util.control.Breaks._

import scala.language.existentials

object TreeLSTMSentiment {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def main(args: Array[String]): Unit = {
    val localParser = new OptionParser[TreeLSTMSentimentParam]("TreeLSTM Sentiment") {
      opt[String]('b', "baseDir")
        .required()
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
//      opt[String]('p', "partitionNum")
//        .text("you may want to tune the partitionNum if run into spark mode")
//        .action((x, c) => c.copy(partitionNum = x.toInt))
//      opt[String]('s', "maxSequenceLength")
//        .text("maxSequenceLength")
//        .action((x, c) => c.copy(maxSequenceLength = x.toInt))
//      opt[String]('w', "maxWordsNum")
//        .text("maxWordsNum")
//        .action((x, c) => c.copy(maxWordsNum = x.toInt))
//      opt[String]('l', "trainingSplit")
//        .text("trainingSplit")
//        .action((x, c) => c.copy(trainingSplit = x.toDouble))
      opt[String]('z', "batchSize")
        .text("batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[String]('h', "hiddenSize")
        .text("batchSize")
        .action((x, c) => c.copy(hiddenSize = x.toInt))
      opt[String]('h', "learingRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
      opt[String]('e', "embLearningRate")
        .text("embedding learning rate")
        .action((x, c) => c.copy(embLearningRate = x.toDouble))
      opt[String]('r', "regRate")
        .text("regularization rate")
        .action((x, c) => c.copy(embLearningRate = x.toDouble))
    }

    localParser.parse(args, TreeLSTMSentimentParam()).map { param =>
      log.info(s"Current parameters: $param")
      val treeLSTMSentiment = new TreeLSTMSentiment(param)
      treeLSTMSentiment.train()
    }
  }

  case class TreeLSTMSentimentParam (
    override val batchSize: Int = 25,
    override val baseDir: String = "./",
    hiddenSize: Int = 150,
    learningRate: Double = 0.05,
    embLearningRate: Double = 0.1,
    regRate: Double = 1e-4,
    fineGrained: Boolean = true,
    dropout: Boolean = true
  ) extends AbstractTextClassificationParams
}

class TreeLSTMSentiment(param: TreeLSTMSentimentParam) extends Serializable{
  import com.intel.analytics.bigdl.numeric.NumericFloat
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val gloveDir = s"${param.baseDir}/glove.6B/"
  val textDataDir = s"${param.baseDir}/20_newsgroup/"
  val classNum = if (param.fineGrained) 5 else 3
  val criterion = ClassNLLCriterion()
  val textClassifier = new TextClassifier(param)

  val treeLSTM = BinaryTreeLSTM(
    inputSize = 10,
    hiddenSize = param.hiddenSize
  )

  def createSentimentModule(): Module[Float] = {
    val sentimentModule = Sequential()

    if (param.dropout) {
      sentimentModule.add(Dropout())
    }

    sentimentModule
      .add(Linear(param.hiddenSize, classNum))
      .add(LogSoftMax())
  }

  def loadRawData(): Unit = {

  }

  def readTree(parents: Array[Int]): Tensor[Float] = {
    val size = parents.length
    val maxNumChildren = parents
      .groupBy(x => x)
      .foldLeft(0)((maxNum, p) => scala.math.max(maxNum, p._2.length))
    val trees = new TensorTree(Tensor[Float](size, maxNumChildren + 1))
    for (i <- parents.indices) {
      if (trees.noChild(i) && parents(i - 1) != -1) {
        var idx = i
        var prev = 0
        while (true) {
          val parent = parents(i - 1)
          if (parent == -1) break
          if (prev != 0) {
            trees.addChild(idx, prev)
          }
          if (trees.hasChild(parent)) {
            trees.addChild(parent, idx)
          } else if (parent == 0) {
            trees.markAsRoot(idx)
            break()
          } else {
            prev = idx
            idx = parent
          }
        }
      }
    }

    var leafIdx = 1
    for (i <- 1 to size) {
      if (trees.noChild(i)) {
        trees.markAsLeaf(i, leafIdx)
        leafIdx += 1
      }
    }

    trees.content
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

    val treeRDD = sc.textFile("", param.partitionNum)
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
    val labelRDD = sc.textFile("", param.partitionNum)
      .map(line => line.split(" "))
      .map(_.map(_.toFloat))
    val sentenceRDD = sc.textFile("", param.partitionNum)


    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
//    val dataRdd = sc.parallelize(loadRawData(), param.partitionNum)
//    val (word2Meta, wordVecMap) = textClassifier.analyzeTexts(dataRdd)

    val filename = s"$gloveDir/glove.840B.300d.txt"
//    val gloveVocab = scala.collection.mutable.Map[String, Int]()
    val word2Vec = scala.collection.mutable.Map[String, Array[Float]]()
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      word2Vec += word -> coefs
    }

    val word2VecTensor = Tensor(word2Vec.size, param.embeddingDim)
    val vocab = scala.collection.mutable.Map[String, Int]()
    var i = 1
    for (line <- Source.fromFile("vocab.cased.txt", "ISO-8859-1").getLines) {
      if (word2Vec.contains(line)) {
        word2VecTensor(i) = Tensor(Storage(word2Vec(line)))
      } else {
        word2VecTensor(i).uniform(-0.05f, 0.05f)
      }
      vocab += line -> i
      i += 1
    }

//    while (i <= wordVecMap.size) {
//      val vector = wordVecMap(i)
//      while (j <= param.embeddingDim) {
//        wordVecTensor.setValue(i, j, vector(j))
//      }
//      i += 1
//    }

    val vocabBC = sc.broadcast(vocab)
    val word2VecBC = sc.broadcast(word2VecTensor)
    val vecSentence = sentenceRDD
      .map(line => line.split(" "))
      .map(line => line.map(vocabBC.value(_)))

    val sampleRDD = vecSentence.zip(labelRDD).zip(treeRDD)
      .map { case ((input: Array[Int], label: Array[Float]), tree: Tensor[Float]) =>
      Sample(
        featureTensor =
          Tensor(input.map(_.toFloat), Array(sequenceLen, embeddingDim))
            .transpose(1, 2)
            .contiguous(),
        labelTensor =
          Tensor(label, Array(1)))
    }

    val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
      Array(trainingSplit, 1 - trainingSplit))

    val optimizer = Optimizer(
      model = buildModel(vocab.size, word2VecTensor),
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

  def buildModel(vocabSize: Int, word2VecTensor: Tensor[Float]): Module[Float] = {
    val embedding = LookupTable(vocabSize, param.embeddingDim)
    embedding.weight.set(word2VecTensor)
    Sequential()
      .add(ParallelTable()
        .add(embedding)
        .add(Identity()))
      .add(BinaryTreeLSTM(
        param.embeddingDim, param.hiddenSize))
  }
}
