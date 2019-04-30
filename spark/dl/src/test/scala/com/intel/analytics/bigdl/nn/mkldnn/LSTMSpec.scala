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
import com.intel.analytics.bigdl.tensor.{DenseTensor, Tensor}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

class LSTMSpec extends FlatSpec with Matchers{
  /*
  "LSTM UnidirectionalInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](Array(common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to inputSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeight.setValue(a, b, i, j, k, 1.0f)
    /* val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeight")
    // println(initWeight)

    var initWeightIter = Tensor[Float](Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to hiddenSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeightIter.setValue(a, b, i, j, k, 1.0f)
    /* val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeightIter")
    // println(initWeightIter)

    var initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize)
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to lstm_n_gates; j <- 1 to hiddenSize)
      initBias.setValue(a, b, i, j, 1.0f)
    // val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()
    // println("initBias")
    // println(initBias)

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Uni Left2Right\n" + output1)

    direction = Direction.UnidirectionalRight2Left
    val lstm2 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initMemoryDescs(Array(inputFormat))
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output2 = lstm2.forward(input)
    // println("DNN output Uni Right2Left\n" + output2)


    input = input.transpose(1, 2)
    val nn_model = nn.Recurrent().add(nn.LSTM2(inputSize, hiddenSize))
    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Uni Left2Right\n" + nn_output)

    val i2g1: Table = nn_model.getParametersTable().get("i2g1").get
    var i2g1_w: Tensor[Float] = i2g1.get("weight").get
    val i2g1_b: Tensor[Float] = i2g1.get("bias").get

    val i2g2: Table = nn_model.getParametersTable().get("i2g2").get
    var i2g2_w: Tensor[Float] = i2g2.get("weight").get
    val i2g2_b: Tensor[Float] = i2g2.get("bias").get

    val i2g3: Table = nn_model.getParametersTable().get("i2g3").get
    val i2g3_w: Tensor[Float] = i2g3.get("weight").get
    val i2g3_b: Tensor[Float] = i2g3.get("bias").get

    val i2g4: Table = nn_model.getParametersTable().get("i2g4").get
    val i2g4_w: Tensor[Float] = i2g4.get("weight").get
    val i2g4_b: Tensor[Float] = i2g4.get("bias").get

    val h2g1: Table = nn_model.getParametersTable().get("h2g1").get
    val h2g1_w: Tensor[Float] = h2g1.get("weight").get
    val h2g1_b: Tensor[Float] = h2g1.get("bias").get

    val h2g2: Table = nn_model.getParametersTable().get("h2g2").get
    val h2g2_w: Tensor[Float] = h2g2.get("weight").get
    val h2g2_b: Tensor[Float] = h2g2.get("bias").get

    val h2g3: Table = nn_model.getParametersTable().get("h2g3").get
    val h2g3_w: Tensor[Float] = h2g3.get("weight").get
    val h2g3_b: Tensor[Float] = h2g3.get("bias").get

    val h2g4: Table = nn_model.getParametersTable().get("h2g4").get
    val h2g4_w: Tensor[Float] = h2g4.get("weight").get
    val h2g4_b: Tensor[Float] = h2g4.get("bias").get
  }
  */

  /*
  "LSTM BidirectionalConcatInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.BidirectionalConcat

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    /*
    var input = Tensor(Array(seqLength, batchSize, inputSize))
    for (i <- 1 to seqLength; j <- 1 to batchSize; k <- 1 to inputSize)
      input.setValue(i, j, k, 0.2f)
    */

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](Array(common_n_layers, 2, inputSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to inputSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeight.setValue(a, b, i, j, k, 1.0f)
    /* val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeight")
    // println(initWeight)

    var initWeightIter = Tensor[Float](Array(common_n_layers, 2,
      hiddenSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to hiddenSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeightIter.setValue(a, b, i, j, k, 1.0f)
    /* val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeightIter")
    // println(initWeightIter)

    var initBias = Tensor[Float](common_n_layers, 2, lstm_n_gates, hiddenSize)
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to lstm_n_gates; j <- 1 to hiddenSize)
      initBias.setValue(a, b, i, j, 1.0f)
    // val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()
    // println("initBias")
    // println(initBias)

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Bi Concat\n" + output1)

    input = input.transpose(1, 2)
    val nn_model = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM2(inputSize, hiddenSize))
    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Bi Concat\n" + nn_output)
  }
  */

  /*
  "LSTM BidirectionalSumInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.BidirectionalSum

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    /*
    var input = Tensor(Array(seqLength, batchSize, inputSize))
    for (i <- 1 to seqLength; j <- 1 to batchSize; k <- 1 to inputSize)
      input.setValue(i, j, k, 0.2f)
    */

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](Array(common_n_layers, 2, inputSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to inputSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeight.setValue(a, b, i, j, k, 1.0f)
    /* val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeight")
    // println(initWeight)

    var initWeightIter = Tensor[Float](Array(common_n_layers, 2,
      hiddenSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to hiddenSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeightIter.setValue(a, b, i, j, k, 1.0f)
    /* val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeightIter")
    // println(initWeightIter)

    var initBias = Tensor[Float](common_n_layers, 2, lstm_n_gates, hiddenSize)
    for (a <- 1 to common_n_layers; b <- 1 to 2;
         i <- 1 to lstm_n_gates; j <- 1 to hiddenSize)
      initBias.setValue(a, b, i, j, 1.0f)
    // val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()
    // println("initBias")
    // println(initBias)

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Bi Sum\n" + output1)

    input = input.transpose(1, 2)
    val nn_model = nn.BiRecurrent(merge = nn.CAddTable()).add(nn.LSTM2(inputSize, hiddenSize))
    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Bi Sum\n" + nn_output)
  }
  */

  "LSTM UnidirectionalInferenceWithRandomParams updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    /*
    var initWeight = Tensor[Float](Array(common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to inputSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeight.setValue(a, b, i, j, k, 0.5f)
    /* val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeight")
    // println(initWeight)

    var initWeightIter = Tensor[Float](Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to hiddenSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeightIter.setValue(a, b, i, j, k, 0.6f)
    /* val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeightIter")
    // println(initWeightIter)

    var initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize)
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to lstm_n_gates; j <- 1 to hiddenSize)
      initBias.setValue(a, b, i, j, 0.2f)
    // val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()
    // println("initBias")
    // println(initBias)
    */

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        inputSize, lstm_n_gates, hiddenSize)).rand()

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize)).rand()

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, hiddenSize)).rand()

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Uni Left2Right Random\n" + output1)

    direction = Direction.UnidirectionalRight2Left
    val lstm2 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initMemoryDescs(Array(inputFormat))
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output2 = lstm2.forward(input)
    // println("DNN output Uni Right2Left Random\n" + output2)

    input = input.transpose(1, 2)
    initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
                           .transpose(1, 2)
    initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
                                   .transpose(1, 2)
    initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))

    val nn_model = nn.Recurrent().add(nn.LSTM2(inputSize, hiddenSize))

    val i2g1: Table = nn_model.getParametersTable().get("i2g1").get
    var i2g1_w: Tensor[Float] = i2g1.get("weight").get
    val i2g1_b: Tensor[Float] = i2g1.get("bias").get
    i2g1_w.copy(initWeight(1).transpose(1, 2))
    i2g1_b.copy(initBias(1))

    val i2g2: Table = nn_model.getParametersTable().get("i2g2").get
    var i2g2_w: Tensor[Float] = i2g2.get("weight").get
    val i2g2_b: Tensor[Float] = i2g2.get("bias").get
    i2g2_w.copy(initWeight(3).transpose(1, 2))
    i2g2_b.copy(initBias(3))

    val i2g3: Table = nn_model.getParametersTable().get("i2g3").get
    val i2g3_w: Tensor[Float] = i2g3.get("weight").get
    val i2g3_b: Tensor[Float] = i2g3.get("bias").get
    i2g3_w.copy(initWeight(2).transpose(1, 2))
    i2g3_b.copy(initBias(2))

    val i2g4: Table = nn_model.getParametersTable().get("i2g4").get
    val i2g4_w: Tensor[Float] = i2g4.get("weight").get
    val i2g4_b: Tensor[Float] = i2g4.get("bias").get
    i2g4_w.copy(initWeight(4).transpose(1, 2))
    i2g4_b.copy(initBias(4))

    val h2g1: Table = nn_model.getParametersTable().get("h2g1").get
    val h2g1_w: Tensor[Float] = h2g1.get("weight").get
    val h2g1_b: Tensor[Float] = h2g1.get("bias").get
    h2g1_w.copy(initWeightIter(1).transpose(1, 2))
    h2g1_b.copy(Tensor(hiddenSize).zero())

    val h2g2: Table = nn_model.getParametersTable().get("h2g2").get
    val h2g2_w: Tensor[Float] = h2g2.get("weight").get
    val h2g2_b: Tensor[Float] = h2g2.get("bias").get
    h2g2_w.copy(initWeightIter(3).transpose(1, 2))
    h2g2_b.copy(Tensor(hiddenSize).zero())

    val h2g3: Table = nn_model.getParametersTable().get("h2g3").get
    val h2g3_w: Tensor[Float] = h2g3.get("weight").get
    val h2g3_b: Tensor[Float] = h2g3.get("bias").get
    h2g3_w.copy(initWeightIter(2).transpose(1, 2))
    h2g3_b.copy(Tensor(hiddenSize).zero())

    val h2g4: Table = nn_model.getParametersTable().get("h2g4").get
    val h2g4_w: Tensor[Float] = h2g4.get("weight").get
    val h2g4_b: Tensor[Float] = h2g4.get("bias").get
    h2g4_w.copy(initWeightIter(4).transpose(1, 2))
    h2g4_b.copy(Tensor(hiddenSize).zero())

    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Uni Left2Right Random\n" + nn_output)
  }

}
