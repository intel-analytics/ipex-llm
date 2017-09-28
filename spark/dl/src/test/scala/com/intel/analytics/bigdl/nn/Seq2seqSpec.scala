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

import com.intel.analytics.bigdl.nn.Seq2seq
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class Seq2seqSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Seq2seq forward" should "work " in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
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

    val decoderCells = Array(ConvLSTMPeephole[Double](
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

    val model = Seq2seq(encoderCells, decoderCells, seqLength, broadcastState = true,
    decoderInput = DecoderInput.ENCODERINPUT)
    
    for (i <- 0 until 3) {
      model.forward(input).toTensor
      model.backward(input, gradOutput)
    }

    val model2 = Seq2seq(encoderCells, decoderCells, seqLength, broadcastState = true,
      decoderInput = DecoderInput.ENCODEROUTPUTSTATE)

    for (i <- 0 until 3) {
      val output = model2.forward(input).toTensor
      model2.backward(input, gradOutput)
    }

    val model3 = Seq2seq(encoderCells, decoderCells, seqLength, broadcastState = true,
      decoderInput = DecoderInput.ZEROS)

    for (i <- 0 until 3) {
      val output = model3.forward(input).toTensor
      model3.backward(input, gradOutput)
    }
  }
}

