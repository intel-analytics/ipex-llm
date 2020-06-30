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
import scala.reflect.ClassTag
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
            var parent =
              if (idx != 0) parents(idx - 1)
              else -1
            if (parent == parents.length) parent = 0
            if (prev != 0 && parent != -1) {
              trees.addChild(idx + 1, prev + 1)
            }

            if (parent == -1) {
              trees.markAsRoot(1)
              if (prev != 0) {
                trees.addChild(1, prev + 1)
              }
              break()
            } else if (trees.hasChild(parent + 1)) {
              trees.addChild(parent + 1, idx + 1)
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
    for (i <- 2 to size) {
      if (trees.noChild(i)) {
        trees.markAsLeaf(i, leafIdx)
        leafIdx += 1
      }
    }

    trees.content
  }

  def remapLabel(
    label: Float
  ): Float = {
    label + 3
  }

  /**
   * Rotate an array `arr` from `offset` distance to the end
   *
   * @param arr Given array
   * @param offset right rotate how many elements
   */
  def rotate[D](arr: Array[D], offset: Int): Array[D] = {
    if (arr == null || arr.length==0 || offset < 0) {
      throw new IllegalArgumentException("Illegal argument!")
    }

    val newOffset = if (offset > arr.length) offset % arr.length else offset

    val index = arr.length - newOffset

    reverse(arr, 0, index - 1)
    reverse(arr, index, arr.length - 1)
    reverse(arr, 0, arr.length - 1)

    arr
  }

  def reverse[D](arr: Array[D], l: Int, r: Int): Unit = {
    var left = l
    var right = r

    if(arr == null || arr.length == 1) return

    while(left < right) {
      val temp = arr(left)
      arr(left) = arr(right)
      arr(right) = temp
      left += 1
      right -= 1
    }
  }

  def preProcessData(
    sc: SparkContext,
    vocabBC: Broadcast[Map[String, Int]],
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
      .map(_.map(l => remapLabel(l.toFloat)))
      .map(line => rotate(line, 1))
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
    def indexAndSort[D: ClassTag, P <: Product2[Long, D]](rdd: RDD[D]) = {
      rdd.zipWithIndex.map(r => r.swap).sortByKey()
    }

    indexAndSort(sentenceRDD)
      .join(indexAndSort(labelRDD))
      .join(indexAndSort(treeRDD))
      .values
      .map{ case ((input, label), tree) =>
        Sample(
          featureTensors =
            Array(Tensor(input.map(_.toFloat), Array(input.length, 1)),
              tree.resize(tree.size())),
          labelTensor =
            Tensor(label, Array(label.length)))
      }
  }

  def loadEmbeddingAndVocabulary(
    sc: SparkContext,
    w2vPath: String,
    vocabPath: String,
    indexFrom: Int
  ):
  (Tensor[Float], Map[String, Int]) = {
    val word2Vec = sc.textFile(w2vPath)
      .map(line => {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      word -> coefs
    }).toLocalIterator.toList.toMap

    var i = 1
    val vocabLines = sc.textFile(vocabPath).collect()
    val word2VecTensor =
      Tensor(vocabLines.length + indexFrom - 1, word2Vec.last._2.length)

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
    opt[String]('i', "batchSize")
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
    opt[String]('p', "p")
      .text("dropout rate")
      .action((x, c) => c.copy(p = x.toDouble))
    opt[String]('e', "epoch")
      .text("max epoch")
      .action((x, c) => c.copy(epoch = x.toInt))
    opt[String]("optimizerVersion")
      .text("state optimizer version")
      .action((x, c) => c.copy(optimizerVersion = Some(x)))
  }

  case class TreeLSTMSentimentParam (
    override val baseDir: String = "/tmp/.bigdl/dataset/",
    override val batchSize: Int = 128,
    hiddenSize: Int = 250,
    override val learningRate: Double = 0.05,
    regRate: Double = 1e-4,
    p: Double = 0.5,
    epoch: Int = 5,
    optimizerVersion: Option[String] = None
  ) extends AbstractTextClassificationParams
}
