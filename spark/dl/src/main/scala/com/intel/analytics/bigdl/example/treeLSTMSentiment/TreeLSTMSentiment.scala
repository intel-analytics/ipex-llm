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
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
// import com.intel.analytics.bigdl.example.utils._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.LoggerFilter
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser
import scala.util.control.Breaks._

import scala.language.existentials

object TreeLSTMSentiment {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def main(args: Array[String]): Unit = {
    val localParser = new OptionParser[TreeLSTMSentimentParam]("TreeLSTM Sentiment") {
      opt[String]('b', "baseDir")
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

    localParser.parse(args, TreeLSTMSentimentParam()).foreach { param =>
      log.info(s"Current parameters: $param")
      val treeLSTMSentiment = new TreeLSTMSentiment(param)
      treeLSTMSentiment.train()
    }
  }

  case class TreeLSTMSentimentParam (
    override val batchSize: Int = 24,
    override val baseDir: String = "/tmp/.bigdl/dataset/",
    hiddenSize: Int = 150,
    learningRate: Double = 0.05,
    embLearningRate: Double = 0.1,
    regRate: Double = 1e-4,
    fineGrained: Boolean = true,
    dropout: Boolean = true
  ) extends AbstractTextClassificationParams
}

class TreeLSTMSentiment(param: TreeLSTMSentimentParam) extends Serializable {
  import com.intel.analytics.bigdl.numeric.NumericFloat
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val DATA_DIR = param.baseDir
  val classNum = if (param.fineGrained) 5 else 3
  val criterion = ClassNLLCriterion()
  val textClassifier = new TextClassifier(param)

  val treeLSTM = BinaryTreeLSTM(
    inputSize = 10,
    hiddenSize = param.hiddenSize
  )

  val readTree = (parents: Array[Int]) => {
    val size = parents.length
    val maxNumChildren = parents
      .groupBy(x => x)
      .foldLeft(0)((maxNum, p) => scala.math.max(maxNum, p._2.length))
    val trees = new TensorTree(Tensor[Float](size, maxNumChildren + 1))
    for (i <- parents.indices) {
      if (trees.noChild(i + 1) && parents(i) != -1) {
        var idx = i + 1
        var prev = 0
        breakable {
          while (true) {
            val parent = parents(idx - 1)
            if (parent == -1) break
            if (prev != 0) {
              trees.addChild(idx, prev)
            }
            if (trees.exists(parent)) {
              trees.addChild(parent, idx)
            } else if (parent == 0) {
              trees.markAsRoot(idx)
              break
            } else {
              prev = idx
              idx = parent
            }
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
    val trainingSplit = param.trainingSplit

    val treeRDD = sc.textFile(s"$DATA_DIR/sst/train/parents.txt", param.partitionNum)
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
    print(s"treeRDD count: ${treeRDD.count()}")
    val labelRDD = sc.textFile(s"$DATA_DIR/sst/train/labels.txt", param.partitionNum)
      .map(line => line.split(" "))
      .map(_.map(_.toFloat))
    print(s"labelRDD count: ${labelRDD.count()}")
    val sentenceRDD = sc.textFile(s"$DATA_DIR/sst/train/sents.txt", param.partitionNum)
    print(s"sentenceRDD count: ${sentenceRDD.count()}")



    // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
//    val dataRdd = sc.parallelize(loadRawData(), param.partitionNum)
//    val (word2Meta, wordVecMap) = textClassifier.analyzeTexts(dataRdd)

    val gloveDir = s"$DATA_DIR/glove/glove.840B.300d.txt"
    val gloveVocab = scala.collection.mutable.Map[String, Int]()
    val word2Vec = scala.collection.mutable.Map[String, Array[Float]]()
    for (line <- Source.fromFile(gloveDir, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      word2Vec += word -> coefs
    }

    val vocab = scala.collection.mutable.Map[String, Int]()

    val paddingValue = 1
    val oovChar = 2
    val indexFrom = 3
    var i = 1
    val vocabLines = Source.fromFile(s"$DATA_DIR/sst/vocab-cased.txt", "ISO-8859-1").getLines.toList
    val word2VecTensor = Tensor(vocabLines.length, 300)
    for (line <- vocabLines) {
      if (i < indexFrom || !word2Vec.contains(line)) {
        word2VecTensor.select(1, i).apply1(_ => RNG.uniform(-0.05f, 0.05f).toFloat)
      } else {
        word2VecTensor.select(1, i).copy(Tensor(Storage(word2Vec(line))))
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
    val vecSentence: RDD[Array[Int]] = sentenceRDD
      .map(line => line.split(" "))
      .map(line => line.map(vocabBC.value.getOrElse(_, oovChar)))
    print(s"vecSentence count: ${vecSentence.count()}")

    def indexAndSort(rdd: RDD[_]) = rdd.zipWithIndex.map(_.swap).sortByKey()

    val sampleRDD = indexAndSort(vecSentence)
      .join(indexAndSort(labelRDD))
      .join(indexAndSort(treeRDD))
      .values
      .map { case ((input: Array[Int], label: Array[Float]), tree: Tensor[Float]) =>
      Sample(
        featureTensors =
          Array(Tensor(input.map(_.toFloat), Array(input.length, 1)),
        tree.resize(tree.size() ++ Array(1))),
        labelTensor =
          Tensor(label, Array(1)))
    }

    val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
      Array(trainingSplit, 1 - trainingSplit))
    trainingRDD.count()

    val optimizer = Optimizer(
      model = buildModel(vocab.size, word2VecTensor),
      sampleRDD = trainingRDD,
      criterion = criterion,
      batchSize = param.batchSize,
      isInOrder = false,
      featurePaddings = Some(Array(Tensor(T(paddingValue.toFloat)), Tensor(T(-1f)))),
      labelPadding = Some(-1f)
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
      .add(MapTable(Squeeze(3)))
      .add(ParallelTable()
        .add(embedding)
        .add(Identity()))
      .add(BinaryTreeLSTM(
        param.embeddingDim, param.hiddenSize))
  }
}
