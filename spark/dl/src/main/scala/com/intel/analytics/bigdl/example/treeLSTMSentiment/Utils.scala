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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.example.utils.AbstractTextClassificationParams
import com.intel.analytics.bigdl.nn.TensorTree
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import scopt.OptionParser

import scala.io.Source
import scala.language.existentials
import scala.util.control.Breaks._

object Utils {
  def readTree(
    parents: Array[Int]
  ): Tensor[Float] = {
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

            if (parent == 0) {
              trees.markAsRoot(idx)
              break()
            } else if (trees.hasChild(parent)) {
              trees.addChild(parent, idx)
              break()
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

  def remapLabel(
    label: Float,
    fineGrained: Boolean
  ): Float = {
    if (fineGrained) {
      label + 3
    } else {
      if (label < 0) {
        1f
      } else if (label == 0) {
        2f
      } else {
        3f
      }
    }
  }

  def preProcessData(
    sc: SparkContext,
    vocabBC: Broadcast[Map[String, Int]],
    fineGrained: Boolean,
    oovChar: Int,
    treePath: String,
    labelPath: String,
    sentencePath: String
  ): (RDD[Tensor[Float]], RDD[Array[Float]], RDD[Array[Int]]) = {
    val treeRDD = sc.textFile(treePath, 4)
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
    val labelRDD = sc.textFile(labelPath, 4)
      .map(line => line.split(" "))
      .map(_.map(l => remapLabel(l.toFloat, fineGrained)))
    val sentenceRDD = sc.textFile(sentencePath, 4)
      .map(line => line.split(" "))
      .map(line => line.map(vocabBC.value.getOrElse(_, oovChar)))

    (treeRDD, labelRDD, sentenceRDD)
  }

  def toSample(
    treeRDD: RDD[Tensor[Float]],
    labelRDD: RDD[Array[Float]],
    sentenceRDD: RDD[Array[Int]]
  ): RDD[Sample[Float]] = {
    def indexAndSort(rdd: RDD[_]) = rdd.zipWithIndex.map(_.swap).sortByKey()

    indexAndSort(sentenceRDD)
      .join(indexAndSort(labelRDD))
      .join(indexAndSort(treeRDD))
      .values
      .map { case ((input: Array[Int], label: Array[Float]), tree: Tensor[Float]) =>
        Sample(
          featureTensors =
            Array(Tensor(input.map(_.toFloat), Array(input.length, 1)),
              tree.resize(tree.size())),
          labelTensor =
            Tensor(label, Array(label.length)))
      }
  }

  def loadEmbeddingAndVocabulary(
    w2vPath: String,
    vocabPath: String,
    indexFrom: Int
  ):
  (Tensor[Float], Map[String, Int]) = {
    val word2Vec = scala.collection.mutable.Map[String, Array[Float]]()
    for (line <- Source.fromFile(w2vPath, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      word2Vec += word -> coefs
    }

    var i = 1
    val vocabLines = Source
      .fromFile(vocabPath, "ISO-8859-1")
      .getLines
      .toList
    val word2VecTensor = Tensor(vocabLines.length + indexFrom - 1, word2Vec.last._2.length)

    val vocab = scala.collection.mutable.Map[String, Int]()
    while (i < indexFrom) {
      word2VecTensor.select(1, i).apply1(_ => RNG.uniform(-0.05f, 0.05f).toFloat)
      i += 1
    }

    for (line <- vocabLines) {
      if (!word2Vec.contains(line)) {
        word2VecTensor.select(1, i).apply1(_ => RNG.uniform(-0.05f, 0.05f).toFloat)
      } else {
        word2VecTensor.select(1, i).copy(Tensor(Storage(word2Vec(line))))
      }
      vocab += line -> i
      i += 1
    }

    (word2VecTensor, vocab.toMap)
  }

  val paramParser = new OptionParser[TreeLSTMSentimentParam]("TreeLSTM Sentiment") {
      opt[String]('b', "baseDir")
        .text("Base dir containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]('b', "batchSize")
        .text("batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[String]('h', "hiddenSize")
        .text("hiddenSize")
        .action((x, c) => c.copy(hiddenSize = x.toInt))
      opt[String]('l', "learingRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
      opt[String]('r', "regRate")
        .text("regularization rate")
        .action((x, c) => c.copy(regRate = x.toDouble))
    }

  case class TreeLSTMSentimentParam (
    override val baseDir: String = "/tmp/.bigdl/dataset/",
    override val batchSize: Int = 64,
    hiddenSize: Int = 150,
    learningRate: Double = 0.1,
    regRate: Double = 1e-4,
    fineGrained: Boolean = true,
    dropout: Boolean = true
  ) extends AbstractTextClassificationParams
}
