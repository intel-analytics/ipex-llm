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


package com.intel.analytics.bigdl.nn.mkldnn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory, RNNCellFlags}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

class LSTMSpec extends FlatSpec with Matchers{
  "LSTM UnidirectionalLeft2Right updateOutput" should "work correctly" in {
    val seqLength = 4
    val batchSize = 2
    val inputSize = 4

    val f = AlgKind.EltwiseTanh
    val flags = RNNCellFlags.RNNCellWithRelu
    val alpha = 0F
    val clipping = 0F
    val direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    // val input = Tensor[Float]().resize(seqLength, batchSize, inputSize).rand()
    val input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, inputSize).rand()
    val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, inputSize).rand()

    val lstm = LSTM(inputSize, f, flags, alpha, clipping, direction, initWeight, initBias)
    lstm.setRuntime(new MklDnnRuntime)
    lstm.reset()
    lstm.initMemoryDescs(Array(inputFormat))
    lstm.initFwdPrimitives(Array(inputFormat), InferencePhase)

    val output = lstm.forward(input)
    println("DNN output \n" + output)

    val nnlstm = nn.Recurrent().add(nn.LSTM(inputSize, inputSize,
      activation = nn.Tanh(), innerActivation = nn.Tanh()))

    val nnoutput = nnlstm.forward(input)
    println("NN output \n" + nnoutput)
  }
}
