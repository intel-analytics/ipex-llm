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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

class SoftMaxSpec extends FlatSpec with Matchers {
  "SoftMax forward 1-D" should "work correctly" in {
    // we should test the cases which contain 1
    val tests = List(2, 1)

    for (x <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(x), Memory.Format.x)), TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(x), Memory.Format.x)), TrainingPhase)

      val input = Tensor(x).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) should be (nnOutput)

      sm.backward(input, nnOutput)
      nnSm.backward(input, nnOutput)

      Tools.dense(sm.gradInput) should be (nnSm.gradInput)
    }
  }

  "SoftMax forward 2-D" should "work correctly" in {
    val tests = List(
      (2, 3),
      (1, 3),
      (1, 1),
      (2, 1))

    for ((batchSize, channel) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel), Memory.Format.nc)),
        TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(batchSize, channel), Memory.Format.nc)),
        TrainingPhase)

      val input = Tensor(batchSize, channel).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) shouldEqual nnOutput

      sm.backward(input, nnOutput)
      nnSm.backward(input, nnOutput)

      Tools.dense(sm.gradInput) should be (nnSm.gradInput)
    }
  }

  "SoftMax forward 4-D" should "work correctly" in {
    // we should test the cases which contain 1
    val tests = List(
      (2, 3, 4, 4),
      (1, 3, 4, 4),
      (1, 3, 1, 1),
      (1, 1, 1, 1),
      (1, 1, 3, 3),
      (2, 1, 3, 3),
      (2, 2, 1, 1))

    for ((batchSize, channel, height, width) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
        Memory.Format.nchw)), TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
        Memory.Format.nchw)), TrainingPhase)

      val input = Tensor(batchSize, channel, height, width).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) should be (nnOutput)

      sm.backward(input, nnOutput)
      nnSm.backward(input, nnOutput)

      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnSm.gradInput.toTensor,
        epsilon = 10-5)
    }
  }

  "SoftMax forward 3-D" should "work correctly" in {
    // we should test the cases which contain 1
    val tests = List(
      (3, 4, 4),
      (3, 4, 4),
      (3, 1, 1),
      (1, 1, 1),
      (1, 3, 3),
      (1, 3, 3),
      (2, 1, 1))

    for ((i, j, k) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(i, j, k), Memory.Format.ncw)), TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(i, j, k), Memory.Format.ncw)), TrainingPhase)

      val input = Tensor(i, j, k).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) should be (nnOutput)
      sm.backward(input, nnOutput)
      nnSm.backward(input, nnOutput)

      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnSm.gradInput.toTensor,
        epsilon = 10-5)
    }
  }

  "SoftMax backward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), TrainingPhase)
    sm.initBwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), TrainingPhase)

    val nnSm = nn.SoftMax()

    val input = Tensor(batchSize, channel, height, width).rand()
    val gradOutput = Tensor().resizeAs(input).rand(-10, 10)

    sm.forward(input)
    nnSm.forward(input)

    sm.backward(input, gradOutput)
    nnSm.backward(input, gradOutput)

    Equivalent.nearequals(Tools.dense(sm.output).toTensor, nnSm.output.toTensor,
      epsilon = 10-4)
    Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnSm.gradInput.toTensor,
      epsilon = 10-4)
  }

  "SoftMax multi times forward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), InferencePhase)
    sm.evaluate()

    val nnSm = nn.SoftMax()

    (0 until 5).foreach { _ =>
      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      sm.forward(input)
      nnSm.forward(input)

      Tools.dense(sm.output) should be (nnSm.output)
    }
  }

  "axis" should "work correctly" in {
    val input = Tensor[Float](2, 24564, 21).rand(-1, 1)

    val sm1 = SoftMax(axis = 2)
    val seq1 = Sequential()
      .add(Input(Array(2, 24564, 21), Memory.Format.ntc))
      .add(sm1)
      .add(Output(Memory.Format.ntc))
    seq1.asInstanceOf[MklDnnContainer].compile(TrainingPhase)

    seq1.forward(input)

    input.resize(Array(2 * 24564, 21))

    val sm2 = SoftMax()
    val seq2 = Sequential().add(Input(Array(2 * 24564, 21), Memory.Format.nc))
        .add(sm2)
        .add(Output())
    seq2.asInstanceOf[MklDnnContainer].compile(TrainingPhase)
    sm2.evaluate()

    seq2.forward(input)

    seq1.output.toTensor.view(Array(2 * 24564, 21)) should be (seq2.output)
  }

  "softmax with java serialization" should "work correctly" in {
    val inputShape = Array(2, 3, 4, 4)

    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    sm.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    val cloned = SerializationUtils.clone(sm)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    val input = Tensor(inputShape).rand(-1, 1)
    val gradOutput = Tensor(inputShape).rand(-1, 1)

    sm.forward(input)
    cloned.forward(input)

    Tools.dense(sm.output) should be (Tools.dense(cloned.output))

    sm.backward(input, gradOutput)
    cloned.backward(input, gradOutput)

    Tools.dense(sm.gradInput) should be (Tools.dense(cloned.gradInput))

  }

  "SoftMax with 2dims input" should "be ok" in {
    val input = Tensor[Float](T(
      T(-0.33185136,	-0.36650753,	-0.18259251,	-0.28977787,	0.12433326,	-0.2162494,	0.10134846,	-0.3177442,	-0.1484699,	0.13634288),
      T(-0.5104831,	-0.5519625,	-0.11421487,	-0.2595594,	0.16804607,	-0.23292251,	0.07044585,	-0.44675964,	0.12295306,	0.18260688),
      T(-0.48692894,	-0.49688655,	-0.18367237,	-0.27386612,	0.16401377,	-0.18842852,	0.07758184,	-0.31743416,	-0.105054736,	0.16071126)
    ))

    val gradOutput = Tensor[Float](T(
      T(0.9667894,	0.24395128,	0.76337516,	0.9502087,	0.23601186,	0.076125905,	0.62328607,	0.770936,	0.975731,	0.21108276),
      T(0.34206265,	0.8998105,	0.73317754,	0.3519888,	0.6322726,	0.47116464,	0.67990524,	0.7170129,	0.41524208,	0.20622952),
      T(0.3923543,	0.017130794,	0.20689869,	0.37415645,	0.62618166,	0.58558035,	0.87812847,	0.4298241,	0.01781603,	0.029151212)))

    val gradInput = Tensor[Float](T(
    T(0.033650335,	-0.02460857,	0.019749852,	0.033681773,	-0.041227575,	-0.044008553,	0.008562913,	0.017880393,	0.041301828,	-0.04498242),
    T(-0.012584607,	0.024181157,	0.020681644,	-0.015309532,	0.013951356,	-0.0050649773,	0.018423514,	0.013663444,	-0.014368655,	-0.043573305),
    T(0.0013979464,	-0.024840426,	-0.015835835,	1.402814E-4,	0.034327768,	0.020268412,	0.06276163,	0.0047896877,	-0.036685247,	-0.046324227)))

  }
}
