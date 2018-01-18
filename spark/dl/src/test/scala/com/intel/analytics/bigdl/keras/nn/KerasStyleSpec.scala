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

import com.intel.analytics.bigdl.nn.keras.{Dense, Input, Model, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.{Input => TInput, Sequential => TSequential, _}
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

  "Sequential: Dense" should "works correctly" in {
    val seq = KSequential[Float]()
    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
    val d2 = new Dense[Float](5).setName("dense2")
    val d3 = new Dense[Float](6).setName("dense4")
    seq.add(d1)
    seq.add(d2)
    seq.add(d3)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(d3.getOutputShape().toSingle().sameElements(Array(6)))
    require(d3.getInputShape().toSingle().sameElements(Array(5)))
  }

  "Frozen sequential" should "be tested" in {
    intercept[RuntimeException] {
      val seq = KSequential[Float]()
      val seq1 = KSequential[Float]()
      seq.add(seq1)
      seq1.add(Dense[Float](20, inputShape = Array(10)))
    }
  }

  "Sequential: shared relu" should "works correctly" in {
    val sharedRelu = ReLU[Float]()
    val seq1 = KSequential[Float]()
    seq1.add(Dense[Float](20, inputShape = Array(10)))
    seq1.add(sharedRelu)
    require(seq1.getOutputShape().toSingle().sameElements(Array(20)))

    val seq2 = KSequential[Float]()
    seq2.add(Dense[Float](5, inputShape = Array(20)))
    seq2.add(sharedRelu)
    require(seq2.getOutputShape().toSingle().sameElements(Array(5)))

    val seq = KSequential[Float]()
    seq.add(seq1)
    seq.add(seq2)

    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(seq.getInputShape().toSingle().sameElements(Array(10)))
    require(seq.getOutputShape().toSingle().sameElements(Array(5)))
  }


  "TSequential" should "not works with dense" in {
    intercept[RuntimeException] {
      val seq = TSequential[Float]()
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
      seq.add(d1)
    }
  }

  "TGraph" should "not works with dense" in {
    intercept[RuntimeException] {
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1").inputs(Input())
      val l1 = new Linear(2, 3).inputs(d1)
    }
  }

  "TSequential" should "not works with container containing Dense" in {
    val seq = TSequential[Float]()
    intercept[RuntimeException] {
      val parallelTable = ParallelTable[Float]()
      val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
      parallelTable.add(d1)
      seq.add(parallelTable)
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

  // TODO: save Shape
  //  "save and reload model" should "works correctly" in {
  //    val input = Input[Float](inputShape = Array(10))
  //    val d = new Dense[Float](20).setName("dense1").inputs(input)
  //    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
  //    val graph = Model[Float](input, d2)
  //    val tmpFile = createTmpFile()
  //    val absPath = tmpFile.getAbsolutePath
  //    tmpFile.delete()
  //    graph.saveModule(absPath)
  //    val reloadedModel = Module.loadModule(absPath)
  //    val inputData = Tensor[Float](Array(20, 10)).rand()
  //    val output = reloadedModel.forward(inputData)
  //  }
  //



}
