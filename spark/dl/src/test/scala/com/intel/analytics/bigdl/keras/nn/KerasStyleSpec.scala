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

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.nn.{Input => TInput, Sequential => TSequential, _}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.Input._
import com.intel.analytics.bigdl.nn.keras.{Dense, Model, Input, Sequential => KSequential}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper


class KerasStyleSpec extends BigDLSpecHelper {

  "Graph: Dense" should "works correctly" in {
    val input = Input[Float](inputShape = Array(10))
    val d0 = new Dense[Float](20, activation = ReLU()).setName("dense1")
      val d = d0.inputs(input)
    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = model.forward(inputData)
  }

//  "Sequential: Dense" should "works correctly" in {
//    //    intercept[RuntimeException] {
//
//    val seq = TSequential[Float]()
//    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
//    val d2 = new Dense[Float](5).setName("dense2")
//    val d3 = new Dense[Float](6).setName("dense4")
//    seq.add(d1)
//    seq.add(d2)
//    seq.add(d3)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = seq.forward(inputData)
//    require(d3.getOutputShape().toSingle().sameElements(Array(6)))
//    require(d3.getInputShape().toSingle().sameElements(Array(5)))
//  }

  "TSequential" should "not works with dense" in {
    intercept[RuntimeException] {
      val seq = TSequential[Float]()
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
      seq.add(d1)
    }
  }

  "Graph" should "not works with dense" in {
    intercept[RuntimeException] {
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1").inputs(Input())
      val l1 = new Linear(2, 3).inputs(d1)
    }
  }

  "TSequential" should "not works with container with dense" in {
    intercept[RuntimeException] {
      val seq = TSequential[Float]()
      val seq2 = TSequential[Float]()
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
      seq2.add(d1)
      seq.add(seq2)
    }
  }


//
//  "save and reload model" should "works correctly" in {
//    val input = Input[Float](inputShape = Array(10))
//    val d = new Dense[Float](20).setName("dense1").inputs(input)
//    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
//    // mix with the old layer
//    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3").inputs(d2)
//    val d4 = new Dense[Float](6).setName("dense4").inputs(d3)
//    val graph = Graph[Float](input, d4)
//    val tmpFile = createTmpFile()
//    val absPath = tmpFile.getAbsolutePath
//    tmpFile.delete()
//    graph.saveModule(absPath)
//    val reloadedModel = Module.loadModule(absPath)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = reloadedModel.forward(inputData)
//  }
//
//  "Graph: Dense + Linear" should "works correctly" in {
//    val input = Input[Float](inputShape = Array(10))
//    val d = new Dense[Float](20).setName("dense1").inputs(input)
//    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
//    // mix with the old layer
//    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3").inputs(d2)
//    val d4 = new Dense[Float](6).setName("dense4").inputs(d3)
//    val graph = Graph[Float](input, d4)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = graph.forward(inputData)
//  }
//
//
//  "Sequential: Dense + Linear" should "works correctly" in {
//    val seq = Sequential[Float]()
//    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
//    val d2 = new Dense[Float](5).setName("dense2")
//    // mix with the old layer
//    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3")
//    val d4 = new Dense[Float](6).setName("dense4")
//    seq.add(d1)
//    seq.add(d2)
//    seq.add(d3)
//    seq.add(d4)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = seq.forward(inputData)
//    require(d3.getOutputShape().toTensor[Int].toArray().sameElements(Array(30)))
//    require(d3.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
//  }
//
//  "Sequential: Linear + Dense" should "works correctly" in {
//    val seq = Sequential[Float]()
//    val d0 = InputLayer[Float](inputShape = Array(5))
//    val d1 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense1")
//    val d2 = new Dense[Float](20).setName("dense2")
//    seq.add(d0)
//    seq.add(d1)
//    seq.add(d2)
//    val inputData = Tensor[Float](Array(20, 5)).rand()
//    val output = seq.forward(inputData)
//    require(d2.getOutputShape().toTensor[Int].toArray().sameElements(Array(20)))
//    require(d2.getInputShape().toTensor[Int].toArray().sameElements(Array(30)))
//    require(seq.getOutputShape().toTensor[Int].toArray().sameElements(Array(20)))
//    require(seq.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
//  }
//
//  "Sequential: pure old style" should "works correctly" in {
//    val seq = Sequential[Float]()
//    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).setName("dense1")
//    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).setName("dense2")
//    seq.add(d1)
//    seq.add(d2)
//    val inputData = Tensor[Float](Array(2, 5)).rand()
//    val output = seq.forward(inputData)
//  }
//
//  "Graph: pure old style" should "works correctly" in {
//    val input = Input[Float](inputShape = Array(5))
//    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).inputs(input)
//    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).inputs(d1)
//    val graph = Graph[Float](input, d2)
//    val inputData = Tensor[Float](Array(20, 5)).rand()
//    val output = graph.forward(inputData)
//    require(d2.element.getOutputShape().toTensor[Int].toArray().sameElements(Array(7)))
//    require(d2.element.getInputShape().toTensor[Int].toArray().sameElements(Array(6)))
//    require(graph.getOutputShape().toTensor[Int].toArray().sameElements(Array(7)))
//    require(graph.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
//  }
//
//  "Nested Sequential model" should "works correctly" in {
//    val seq1 = Sequential[Float]()
//    val seq2 = Sequential[Float]()
//    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
//    val d2 = new Dense[Float](5, inputShape = Array(20)).setName("dense2")
//    seq1.add(d1)
//    seq1.add(seq2)
//    seq2.add(d2)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = seq1.forward(inputData)
//    require(d2.getOutputShape().toTensor[Int].toArray().sameElements(Array(5)))
//    require(d2.getInputShape().toTensor[Int].toArray().sameElements(Array(20)))
//  }
//
//  "Nested Sequential model: pure old style" should "works correctly" in {
//    val seq1 = Sequential[Float]()
//    val seq2 = Sequential[Float]()
//    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).setName("dense1")
//    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).setName("dense2")
//    seq1.add(d1)
//    seq1.add(seq2)
//    seq2.add(d2)
//    val inputData = Tensor[Float](Array(20, 5)).rand()
//    val output = seq1.forward(inputData)
//  }
//
//  "Nested Sequential model: pure old style with proper shape" should "works correctly" in {
//    val seq1 = Sequential[Float]()
//    val seq2 = Sequential[Float]()
//    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).setName("dense1")
//    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).setName("dense2")
//    seq1.add(InputLayer(inputShape = Array(5)))
//    seq1.add(d1)
//    seq1.add(seq2)
//    seq2.add(InputLayer(inputShape = Array(6)))
//    seq2.add(d2)
//    val inputData = Tensor[Float](Array(20, 5)).rand()
//    val output = seq1.forward(inputData)
//    require(d2.getOutputShape().toTensor[Int].toArray().sameElements(Array(7)))
//    require(d2.getInputShape().toTensor[Int].toArray().sameElements(Array(6)))
//  }
//
//  "Use Dense within ParallelTable" should "not works correctly" in {
//    val seq = Sequential[Float]()
//    intercept[RuntimeException] {
//      val parallelTable = ParallelTable[Float]()
//      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
//      parallelTable.add(d1)
//      seq.add(parallelTable)
//    }
//  }
}
