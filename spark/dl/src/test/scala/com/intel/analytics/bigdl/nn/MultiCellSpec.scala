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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_DOUBLE_TENSOR
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class MultiCellSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A MultiCell " should "work in BatchMode" in {
    val hiddenSize = 5
    val inputSize = 5
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiCell[Double](cells)))

    val input = Tensor[Double](batchSize, inputSize, 10, 10).rand
    val output = model.forward(input).toTensor[Double]
    for (i <- 1 to 3) {
      val output = model.forward(input)
      model.backward(input, output)
    }
  }

//  "A MultiCell " should "generate correct output with convlstm" in {
//    val hiddenSize = 7
//    val inputSize = 7
//    val seqLength = 3
//    val batchSize = 2
//    val kernalW = 3
//    val kernalH = 3
//    val rec = Recurrent[Double]()
//    val cells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Sequential[Double]()
//      .add(rec
//        .add(MultiCell[Double](cells)))
//    val weights = model.getParameters()._1.clone()
//
//    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
//    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
//    val output = model.forward(input).toTensor[Double]
//    val gradInput = model.backward(input, gradOutput).toTensor[Double]
//
//    val model2 = Sequential[Double]()
//      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
//        inputSize,
//        hiddenSize,
//        kernalW, kernalH,
//        1)))
//      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
//        inputSize,
//        hiddenSize,
//        kernalW, kernalH,
//        1)))
//    model2.getParameters()._1.copy(weights)
//
//    val output2 = model2.forward(input).toTensor[Double]
//    val gradInput2 = model2.backward(input, gradOutput).toTensor[Double]
//
//    output.map(output2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//
//    gradInput.map(gradInput2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//  }

//  "A MultiCell " should "generate correct output with lstm" in {
//    val hiddenSize = 10
//    val inputSize = 10
//    val seqLength = 5
//    val batchSize = 2
//    val rec = Recurrent[Double](includeTime = false)
//    val cells = Array(LSTM[Double](
//      inputSize,
//      hiddenSize, includeTime = false),
//      LSTM[Double](
//        inputSize,
//        hiddenSize, includeTime = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Sequential[Double]()
//      .add(rec
//        .add(MultiCell[Double](cells)))
//    val weights = model.getParameters()._1.clone()
//
//    val input = Tensor[Double](batchSize, seqLength, inputSize).rand
//    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
//    val output = model.forward(input).toTensor[Double]
//    val gradInput = model.backward(input, gradOutput).toTensor[Double]
//
//    val model2 = Sequential[Double]()
//      .add(Recurrent[Double]().add(LSTM[Double](
//        inputSize,
//        hiddenSize)))
//      .add(Recurrent[Double]().add(LSTM[Double](
//        inputSize,
//        hiddenSize)))
//    model2.getParameters()._1.copy(weights)
//
//    val output2 = model2.forward(input).toTensor[Double]
//    val gradInput2 = model2.backward(input, gradOutput).toTensor[Double]
//
//    output.map(output2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//
//    gradInput.map(gradInput2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//  }

  "A MultiCell " should "generate correct output with convlstm RecurrentDecoder" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 3
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiCell[Double](cells)))
    val weights = model.getParameters()._1.clone()

    val input = Tensor[Double](batchSize, inputSize, 3, 3).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
    val output = model.forward(input).toTensor[Double]
    val gradInput = model.backward(input, gradOutput).toTensor[Double]

    val input2 = Tensor[Double](Array(batchSize, seqLength, inputSize, 3, 3))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val model2 = Sequential[Double]()
      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
        inputSize,
        hiddenSize,
        kernalW, kernalH,
        1)))
      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
        inputSize,
        hiddenSize,
        kernalW, kernalH,
        1)))
    model2.getParameters()._1.copy(weights)

    val output2 = model2.forward(input2).toTensor[Double]

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MultiCell " should "generate correct output with lstm RecurrentDecoder" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 3
    val batchSize = 2
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiCell[Double](cells)))
    val weights = model.getParameters()._1.clone()

    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize).rand
    val output = model.forward(input).toTensor[Double]
    val gradInput = model.backward(input, gradOutput).toTensor[Double]

    val input2 = Tensor[Double](Array(batchSize, seqLength, inputSize))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val model2 = Sequential[Double]()
      .add(Recurrent[Double]().add(LSTM[Double](
        inputSize,
        hiddenSize)))
      .add(Recurrent[Double]().add(LSTM[Double](
        inputSize,
        hiddenSize)))
    model2.getParameters()._1.copy(weights)

    val output2 = model2.forward(input2).toTensor[Double]

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }
}
