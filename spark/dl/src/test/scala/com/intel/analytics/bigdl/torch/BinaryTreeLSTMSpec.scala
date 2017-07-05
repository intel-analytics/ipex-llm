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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim.{Adagrad, TreeNNAccuracy}
import com.intel.analytics.bigdl.example.treeLSTMSentiment.Utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_FLOAT_TENSOR
import com.intel.analytics.bigdl.utils.{Engine, File, T, Table}
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD
import org.scalatest.time.Millisecond
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.control.Breaks._

class BinaryTreeLSTMSpec extends FlatSpec with Matchers with BeforeAndAfter {
  before {
    Engine.setNodeAndCore(1, 4)
  }
  val readTree = (parents: Array[Int]) => {
    val size = parents.length
    val maxNumChildren = parents
      .groupBy(x => x)
      .foldLeft(0)((maxNum, p) => scala.math.max(maxNum, p._2.length))
    val trees = new TensorTree(Tensor[Double](size, maxNumChildren + 1))
    for (i <- parents.indices) {
      if (trees.noChild(i + 1) && parents(i) != -1) {
        var idx = i + 1
        var prev = 0
        breakable {
          while (true) {
            var parent =
              if (idx != 0) parents(idx - 1)
              else -1
//
//            if (parent == 0) parent = -1
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

//  def readTree(
//    parents: Array[Int]
//  ): Tensor[Double] = {
//    val size = parents.length
//    val maxNumChildren = parents
//      .groupBy(x => x)
//      .foldLeft(0)((maxNum, p) => scala.math.max(maxNum, p._2.length))
//    val trees = new TensorTree(Tensor[Double](size, maxNumChildren + 1))
//    for (i <- parents.indices) {
//      if (trees.noChild(i + 1) && parents(i) != -1) {
//        var idx = i + 1
//        var prev = 0
//        breakable {
//          while (true) {
//            var parent =
//              if (idx != 0) parents(idx - 1)
//              else -1
////
////            if (parent == 0) parent = -1
//            if (parent == parents.length) parent = 0
//            if (prev != 0 && parent != -1) {
//              trees.addChild(idx + 1, prev + 1)
//            }
//
//            if (parent == -1) {
//              trees.markAsRoot(1)
//              if (prev != 0) {
//                trees.addChild(1, prev + 1)
//              }
//              break()
//            } else if (trees.hasChild(parent + 1)) {
//              trees.addChild(parent + 1, idx + 1)
//              break()
//            } else {
//              prev = idx
//              idx = parent
//            }
//          }
//        }
//      }
//    }
//
//    var leafIdx = 1
//    for (i <- 2 to size) {
//      if (trees.noChild(i)) {
//        trees.markAsLeaf(i, leafIdx)
//        leafIdx += 1
//      }
//    }
//
//    trees.content
//  }

  var remapLabel = (label: Double, fineGrained: Boolean) => {
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

  "BinaryTreeLSTM" should "works correctly" in {
    RNG.setSeed(100)
    val criterion = TimeDistributedCriterion(ClassNLLCriterion())
    val embeddingDim = 300
    val vocabLines = Source
      .fromFile(s"/home/yao/git/treelstm/data/sst/vocab-cased.txt", "ISO-8859-1")
      .getLines.toList
    val vocabSize = vocabLines.length
    val emb = File.loadTorch[Tensor[Double]](
      "/tmp/emb.t7")

    val vocab = scala.collection.mutable.Map[String, Int]()
    var i = 1
    for (line <- vocabLines) {
      vocab += line -> i
      i += 1
    }

    val classNum = 5
    val embedding = LookupTable(vocabSize, embeddingDim)
    embedding.weight.copy(emb)
    val treeLSTM = BinaryTreeLSTM(
      embeddingDim, 150, withGraph = false)
    val outputModule = Linear(150, classNum)
    val treeLSTMModule = Sequential()
      .add(treeLSTM)
      .add(TimeDistributed(outputModule))
      .add(TimeDistributed(LogSoftMax()))

    val (parameters, _) = treeLSTMModule.getParameters()
    parameters.copy(File.loadTorch[Tensor[Double]](
      "/tmp/tree.t7"))
//    val tmp = java.io.File.createTempFile("weightTmp", ".t7")
//    val tmpPath = tmp.getAbsolutePath
//    File.saveTorch(parameters, tmpPath, TYPE_FLOAT_TENSOR, true)
    val model = Sequential()
      .add(MapTable(Squeeze(3)))
      .add(ParallelTable()
      .add(embedding)
      .add(Identity()))
      .add(treeLSTMModule)


    val batchSize = 5
    val labelPadding = -1.0
    val paddingValue = 1.0

    def toSample(
      treeRDD: List[Tensor[Double]],
      labelRDD: List[Array[Double]],
      sentenceRDD: List[Array[Int]]
    ): List[Sample[Double]] = {
      sentenceRDD.zip(labelRDD).zip(treeRDD)
        .map { case ((input: Array[Int], label: Array[Double]), tree: Tensor[Double]) =>
          Sample(
            featureTensors =
              Array(Tensor(input.map(_.toDouble), Array(input.length, 1)),
                tree.resize(tree.size())),
            labelTensor =
              Tensor(label, Array(label.length)))
        }
    }

    val DATA_DIR = "/home/yao/git/treelstm/data"
    val trainTrees = Source.fromFile(s"$DATA_DIR/sst/train/parents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
    val trainSentences = Source.fromFile(s"$DATA_DIR/sst/train/sents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(line => line.map(vocab.getOrElse(_, 1)))
    val trainLabels = Source.fromFile(s"$DATA_DIR/sst/train/labels.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(l => remapLabel(l.toDouble, true)))
      .map(line => rotate(line, 1))

    val testTrees = Source.fromFile(s"$DATA_DIR/sst/dev/parents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
    val testSentences = Source.fromFile(s"$DATA_DIR/sst/dev/sents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(line => line.map(vocab.getOrElse(_, 1)))
    val testLabels = Source.fromFile(s"$DATA_DIR/sst/dev/labels.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(l => remapLabel(l.toDouble, true)))
      .map(line => rotate(line, 1))

    val trainList = toSample(trainTrees, trainLabels, trainSentences)
    val testList = toSample(testTrees, testLabels, testSentences)

    val trainData = SampleToMiniBatch(
      batchSize,
      Some(Array(Tensor(T(paddingValue.toDouble)), Tensor(T(-1.0, -1.0, -1.0)))),
      Some(labelPadding),
      None).apply(trainList.toIterator)
    val testData = SampleToMiniBatch(
      batchSize,
      Some(Array(Tensor(T(paddingValue.toDouble)), Tensor(T(-1.0, -1.0, -1.0)))),
      Some(labelPadding),
      None).apply(testList.toIterator)
    val trainArray = new ArrayBuffer[MiniBatch[Double]]()
    for (batch <- trainData) {
      val newBatch = SerializationUtils.clone(batch)
      trainArray.append(newBatch)
    }
    val testArray = new ArrayBuffer[MiniBatch[Double]]()
    for (batch <- testData) {
      val newBatch = SerializationUtils.clone(batch)
      testArray.append(newBatch)
    }

    val opt = new Adagrad(learningRate = 0.1)
    val state = T("learningRate" -> 0.1)
    val (weights, grad) = model.getParameters()

    println(s"hashcode: ${model.hashCode()}")

    val epoch = 1
    var j = 1

    for (i <- 1 to epoch) {
      println(s"epoch $i")
      j = 1
      for (batch <- trainArray) {
        embedding.weight.copy(emb)
        def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
          model.zeroGradParameters()
          val input = batch.getInput()
          val output = model.forward(input)
          val _loss = criterion.forward(output, batch.getTarget())
          val gradInput = criterion.backward(output, batch.getTarget())
          model.backward(input, gradInput)
          (_loss, grad)
        }

        val loss = opt.optimize(feval, weights, state)._2
        println(s"$j-th loss = ${loss(0)}")
      }
      j = j + batchSize
    }


    var accuracy = 0f
    for (batch <- testArray) {
      val input = batch.getInput()
      val output = model.forward(input)
      accuracy += new TreeNNAccuracy(true)
        .apply(output, batch.getTarget().asInstanceOf[Tensor[Double]]).result()._1
    }
    accuracy /= testArray.length
    println(s"Accuracy: ${accuracy}")
  }
}
