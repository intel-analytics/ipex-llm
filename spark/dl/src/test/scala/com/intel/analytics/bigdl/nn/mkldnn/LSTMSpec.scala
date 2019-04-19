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
    val seqLength = 1
    val batchSize = 1
    val inputSize = 2
    val hiddenSize = 2

    val f = AlgKind.EltwiseTanh
    val flags = RNNCellFlags.RNNCellWithRelu
    val alpha = 0F
    val clipping = 0F
    val direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    val input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize).rand()
    val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand()
    val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()

    val lstm = LSTM(inputSize, hiddenSize, f, flags, alpha, clipping,
      direction, initWeight, initWeightIter, initBias)

    lstm.setRuntime(new MklDnnRuntime)
    lstm.reset()
    lstm.initMemoryDescs(Array(inputFormat))
    lstm.initFwdPrimitives(Array(inputFormat), InferencePhase)

    val output = lstm.forward(input)
    println("DNN output \n" + output)

    /*
    val nnlstm = nn.Recurrent().add(nn.LSTM(inputSize, hiddenSize,
      activation = nn.Tanh(), innerActivation = nn.Tanh()))

    val nnoutput = nnlstm.forward(input)
    println("NN output \n" + nnoutput)

    println(input)
    */
  }

  /*
  "LSTM 5DimSync" should "work correctly" in {
    val s1: Int = 1
    val s2: Int = 1
    val s3: Int = 5
    val s4: Int = 4
    val s5: Int = 5

    val tensorMap1: TensorMMap = new TensorMMap(Array(s1, s2, s3, s4, s5))
    val runtime1 = new MklDnnRuntime

    val inputFormat1 = HeapData(Array(s1, s2, s3, s4, s5), Memory.Format.ldigo)
    val outputFormat1 = NativeData(Array(s1, s2, s3, s4, s5), Memory.Format.ldigo)
    tensorMap1.setMemoryData(inputFormat1, outputFormat1, runtime1)

    val input = Tensor(Array(s1, s2, s3, s4, s5)).rand()
    tensorMap1.dense.copy(input)
    tensorMap1.sync()
    println("ldigo")


    val tensorMap2: TensorMMap = new TensorMMap(Array(s1, s2, s3, s4, s5))
    val runtime2 = new MklDnnRuntime

    val inputFormat2 = HeapData(Array(s1, s2, s3, s4, s5), Memory.Format.ldsnc)
    val outputFormat2 = NativeData(Array(s1, s2, s3, s4, s5), Memory.Format.ldsnc)
    tensorMap2.setMemoryData(inputFormat2, outputFormat2, runtime2)

    tensorMap2.dense.copy(input)
    tensorMap2.sync()
    println("ldsnc")


    val tensorMap3: TensorMMap = new TensorMMap(Array(s1, s2, s4, s5))
    val runtime3 = new MklDnnRuntime

    val inputFormat3 = HeapData(Array(s1, s2, s4, s5), Memory.Format.ldgo)
    val outputFormat3 = NativeData(Array(s1, s2, s4, s5), Memory.Format.ldgo)
    tensorMap3.setMemoryData(inputFormat3, outputFormat3, runtime3)

    val input_4d = Tensor(Array(s1, s2, s4, s5)).rand()
    tensorMap3.dense.copy(input_4d)
    tensorMap3.sync()
    println("ldgo")
  }
  */
}