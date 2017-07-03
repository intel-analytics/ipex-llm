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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim.Adagrad
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_FLOAT_TENSOR
import com.intel.analytics.bigdl.utils.{File, T}
import org.scalatest.time.Millisecond
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source
import scala.util.control.Breaks._

class BinaryTreeLSTMSpec extends FlatSpec with Matchers {
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

    val embedding = LookupTable(vocabSize, embeddingDim)
    embedding.weight.copy(emb)
    val treeLSTM = BinaryTreeLSTM(
      embeddingDim, 150)
    val outputModule = Linear(150, 5)
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
//      .add(MapTable(Squeeze(3)))
      .add(ParallelTable()
      .add(embedding)
      .add(Identity()))
      .add(treeLSTMModule)


    val DATA_DIR = "/home/yao/git/treelstm/data"
    val trees = Source.fromFile(s"$DATA_DIR/sst/train/parents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(_.toInt))
      .map(readTree)
      .map(line => line.resize(Array(1) ++ line.size()))
    val sentences = Source.fromFile(s"$DATA_DIR/sst/train/sents.txt").getLines().toList
      .map(line => line.split(" "))
      .map(line => line.map(vocab.getOrElse(_, 1)))
      .map(line => line.map(_.toDouble))
      .map(line => Tensor(Storage(line)))
      .map(line => line.resize(Array(1) ++ line.size()))
    val labels = Source.fromFile(s"$DATA_DIR/sst/train/labels.txt").getLines().toList
      .map(line => line.split(" "))
      .map(_.map(l => remapLabel(l.toDouble, true)))
      .map(line => Tensor(Storage(line)))
      .map(line => line.resize(Array(1) ++ line.size()))

    val opt = new Adagrad(learningRate = 0.1)
    val state = T("learningRate" -> 0.1)
    val (weights, grad) = model.getParameters()

    println(s"hashcode: ${model.hashCode()}")

    val start = System.nanoTime()
    val epoch = 2
    for (i <- 1 to epoch) {
      println(s"epoch $i")
      for (j <- trees.indices) {
        def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
          model.zeroGradParameters()
          val input = T(sentences(j), trees(j))
          val output = model.forward(input)
          val _loss = criterion.forward(output, labels(j))
          val gradInput = criterion.backward(output, labels(j))
          model.backward(input, gradInput)
          (_loss, grad)
        }

        val loss = opt.optimize(feval, weights, state)._2
        println(s"$j-th loss = ${loss(0)}")
        j
      }
    }
    val end = System.nanoTime()
    println("Time cost: ", (end - start) / 1e6)
  }
}
