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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class ConvLSTMPeephole3DSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A ConvLSTMPeepwhole3D" should " work in BatchMode" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = Recurrent[Double]()
    val model = Sequential[Double]()
      .add(rec
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1, withPeephole = true)))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand

    for (i <- 1 to 3) {
      val output = model.forward(input)
      model.backward(input, output)
    }
  }

  "A ConvLSTMPeepwhole3D" should " return expected hidden and cell state" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = Recurrent[Double]()
    val model = Sequential[Double]()
      .add(rec
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1, withPeephole = true)))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand

    var output: Activity = null
    for (i <- 1 to 3) {
      output = model.forward(input)
      model.backward(input, output)
    }

    val finalState = rec.getFinalStateAndCellStatus()._1

    finalState.map(output.asInstanceOf[Tensor[Double]].select(2, seqLength), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
  }
}
