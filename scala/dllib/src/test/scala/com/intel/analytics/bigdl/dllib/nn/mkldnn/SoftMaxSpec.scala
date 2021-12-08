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

package com.intel.analytics.bigdl.dllib.nn.mkldnn

import com.intel.analytics.bigdl.dllib.integration.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.dllib.nn
import com.intel.analytics.bigdl.dllib.nn.SoftMin
import com.intel.analytics.bigdl.dllib.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.Future
import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Serial
class SoftMaxSpec extends TorchSpec with Matchers {
  @transient
  private var results: Array[Future[Unit]] = null
  private var pos = 1
  var internalOutput = Tensor[Float]()
  var internalGradInput = Tensor[Float]()

  private def getPositiveDimension(input: Tensor[Float]): Int = {
    val inputDim = input.nDimension() // data batch dim
    pos = if (pos <= 0) {
      inputDim + pos
    }
    else pos
    require(1 <= pos && pos <= input.nDimension(),
      s"Invalid position: $pos ." + s"input dimension ${input.nDimension()}")
    pos
  }

  def internalUpdateOutput(input: Tensor[Float]): Tensor[Float] = {
    require(1 <= input.nDimension() && input.nDimension() <= 4,
      "1D, 2D, 3D or 4D tensor expected" +
        s"input dimension ${input.nDimension()}")
    pos = getPositiveDimension(input)
    // get nFrame and stride value based on the input
    val (nFrame, stride) = input.nDimension() - pos match {
      case 0 => (1, 1)
      case 1 => (input.size(pos), 1)
      case 2 => (1, input.size(pos + 1) * input.size(pos + 2))
      case _ => (input.size(pos), input.size(pos + 2) * input.size(pos + 3))
    }

    if (results == null || results.length != nFrame * stride) {
      results = new Array[Future[Unit]](nFrame * stride)
    }
    internalOutput.resizeAs(input)
    SoftMin.updateOutput[Float](input, internalOutput, results, pos)
    internalOutput
  }

  def internalUpdateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float]
  = {
    internalGradInput.resizeAs(internalOutput)
    SoftMin.updateGradInput[Float](input, gradOutput, internalGradInput,
      internalOutput, results, pos)
    internalGradInput
  }

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

      val gradOutput = Tensor[Float]().resizeAs(nnOutput).rand(-10, 10)
      sm.backward(input, gradOutput)
      nnSm.backward(input, gradOutput)

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

      val gradOutput = Tensor[Float]().resizeAs(nnOutput).rand(-10, 10)
      sm.backward(input, gradOutput)
      nnSm.backward(input, gradOutput)

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

//      val nnSm = nn.SoftMax()
//      val nnOutput = nnSm.forward(input)
//
//      Tools.dense(output) should be (nnOutput)

      val nnOutput = internalUpdateOutput(input)

      val gradOutput = Tensor[Float]().resizeAs(output.toTensor).rand(-10, 10)
      sm.backward(input, gradOutput)
//      nnSm.backward(input, gradOutput)
      val nnGradInput = internalUpdateGradInput(input, gradOutput)


      Tools.dense(output) should be (output)
      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnGradInput,
        epsilon = 1e-5) should be (true)
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

//      val nnSm = nn.SoftMax()
//      val nnOutput = nnSm.forward(input)
      val nnOutput = internalUpdateOutput(input)
      Tools.dense(output) should be (nnOutput)

      val gradOutput = Tensor[Float]().resizeAs(output.toTensor).rand(-10, 10)
      sm.backward(input, gradOutput)
//      nnSm.backward(input, gradOutput)
      val nnGradInput = internalUpdateGradInput(input, gradOutput)

      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnGradInput,
        epsilon = 1e-5) should be (true)
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

//    val nnSm = nn.SoftMax()

    val input = Tensor(batchSize, channel, height, width).rand()
    val gradOutput = Tensor().resizeAs(input).rand(-10, 10)

    sm.forward(input)
//    nnSm.forward(input)
    val nnOutput = internalUpdateOutput(input)

    sm.backward(input, gradOutput)
//    nnSm.backward(input, gradOutput)
    val nnGradInput = internalUpdateGradInput(input, gradOutput)

//    Equivalent.nearequals(Tools.dense(sm.output).toTensor, nnSm.output.toTensor,
//      epsilon = 1e-5) should be (true)
//    Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnSm.gradInput.toTensor,
//      epsilon = 1e-5) should be (true)
    Equivalent.nearequals(Tools.dense(sm.output).toTensor, nnOutput,
      epsilon = 1e-5) should be (true)
    Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnGradInput,
      epsilon = 1e-5) should be (true)
  }

  "SoftMax multi times forward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), InferencePhase)
    sm.evaluate()

//    val nnSm = nn.SoftMax()

    (0 until 5).foreach { _ =>
      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      sm.forward(input)
//      nnSm.forward(input)
      val nnOutput = internalUpdateOutput(input)

      Tools.dense(sm.output) should be (nnOutput)
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
}
