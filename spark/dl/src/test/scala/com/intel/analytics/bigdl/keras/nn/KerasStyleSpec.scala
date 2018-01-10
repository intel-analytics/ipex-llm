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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import org.scalatest.{FlatSpec, Matchers}


class KerasStyleSpec extends BigDLSpecHelper {

  "save model" should "works correctly" in {
    val input = Input[Float](inputShape = Array(10))
    val d = new Dense[Float](20).setName("dense1").inputs(input)
    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
    // mix with the old layer
    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3").inputs(d2)
    val d4 = new Dense[Float](6).setName("dense4").inputs(d3)
    val graph = Graph[Float](input, d4)
    val tmpFile = createTmpFile()
    graph.saveModule(tmpFile.getAbsolutePath)
//    val reloadedModel = Module.loadModule(tmpFile.getAbsolutePath)
//    val inputData = Tensor[Float](Array(20, 10)).rand()
//    val output = reloadedModel.forward(inputData)
  }

  "Graph: Dense + Linear" should "works correctly" in {
    val input = Input[Float](inputShape = Array(10))
    val d = new Dense[Float](20).setName("dense1").inputs(input)
    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
    // mix with the old layer
    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3").inputs(d2)
    val d4 = new Dense[Float](6).setName("dense4").inputs(d3)
    val graph = Graph[Float](input, d4)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = graph.forward(inputData)
  }


  "Sequential: Dense + Linear" should "works correctly" in {
    val seq = Sequential[Float]()
    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
    val d2 = new Dense[Float](5).setName("dense2")
    // mix with the old layer
    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3")
    val d4 = new Dense[Float](6).setName("dense4")
    seq.add(d1)
    seq.add(d2)
    seq.add(d3)
    seq.add(d4)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(d3.getBatchOutputShape().toTensor[Int].toArray().sameElements(Array(-1, 30)))
    require(d3.getBatchInputShape().toTensor[Int].toArray().sameElements(Array(-1, 5)))
  }

  "Sequential: Linear + Dense" should "works correctly" in {
    val seq = Sequential[Float]()
    val d0 = InputLayer[Float](inputShape = Array(5))
    val d1 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense1")
    val d2 = new Dense[Float](20).setName("dense2")
    seq.add(d0)
    seq.add(d1)
    seq.add(d2)
    val inputData = Tensor[Float](Array(20, 5)).rand()
    val output = seq.forward(inputData)
    require(d2.getBatchOutputShape().toTensor[Int].toArray().sameElements(Array(-1, 20)))
    require(d2.getBatchInputShape().toTensor[Int].toArray().sameElements(Array(-1, 30)))
    require(seq.getBatchOutputShape().toTensor[Int].toArray().sameElements(Array(-1, 20)))
    require(seq.getBatchInputShape().toTensor[Int].toArray().sameElements(Array(-1, 5)))
  }

  "Sequential: pure old style" should "works correctly" in {
    val seq = Sequential[Float]()
    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).setName("dense1")
    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).setName("dense2")
    seq.add(d1)
    seq.add(d2)
    val inputData = Tensor[Float](Array(2, 5)).rand()
    val output = seq.forward(inputData)
  }

  "Graph: pure old style" should "works correctly" in {
    val input = Input[Float](inputShape = Array(5))
    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).inputs(input)
    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).inputs(d1)
    val graph = Graph[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 5)).rand()
    val output = graph.forward(inputData)
    require(d2.element.getBatchOutputShape().toTensor[Int].toArray().sameElements(Array(-1, 7)))
    require(d2.element.getBatchInputShape().toTensor[Int].toArray().sameElements(Array(-1, 6)))
    require(graph.getBatchOutputShape().toTensor[Int].toArray().sameElements(Array(-1, 7)))
    require(graph.getBatchInputShape().toTensor[Int].toArray().sameElements(Array(-1, 5)))
  }
}
