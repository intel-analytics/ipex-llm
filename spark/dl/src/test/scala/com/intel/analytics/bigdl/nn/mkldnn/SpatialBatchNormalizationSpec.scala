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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnStorage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Ignore, Matchers}

class SpatialBatchNormalizationSpec extends FlatSpec with Matchers {
  "a simple bn with random input" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val input = Tensor(100, 1, 10, 10).rand(-1, 1)
    val (channel, height, width) = (1, 10, 10)

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(1, 0.0, initWeight = initWeight, initBias = initBias)
    val nnBn = nn.SpatialBatchNormalization(1, 0.0, initWeight = initWeight, initBias = initBias)

    val inputShape = Array(100, 1, 10, 10)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    val out1 = bn.forward(input)
    val out2 = nnBn.forward(input)

    Equivalent.nearequals(Tools.dense(out1).toTensor, out2, 1e-4) should be(true)

    val gradOutput = Tensor[Float]().resizeAs(input).rand()

    bn.backward(input, gradOutput)
    nnBn.backward(input, gradOutput)

    val gradWeight1 = Tools.dense(bn.gradWeightAndBias.native).toTensor
    val gradWeight2 = nnBn.getParameters()._2

    val weight1 = Tools.dense(bn.weightAndBias.native).toTensor
    val weight2 = nnBn.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)

    Equivalent.nearequals(Tools.dense(bn.gradInput).toTensor, nnBn.gradInput) should be (true)
  }

  "batch norm cloned" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val input = Tensor(100, 1, 10, 10).rand(-1, 1)
    val (channel, height, width) = (1, 10, 10)

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(1, 0.0, initWeight = initWeight, initBias = initBias)

    val inputShape = Array(100, 1, 10, 10)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    bn.forward(input)

    val cloned = SerializationUtils.clone(bn)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    cloned.forward(input)

    Tools.dense(bn.output) should be (Tools.dense(cloned.output))

    val gradOutput = Tensor(inputShape).rand(-1, 1)
    bn.backward(input, gradOutput)
    cloned.backward(input, gradOutput)
    Tools.dense(bn.gradInput) should be (Tools.dense(cloned.gradInput))
    Tools.dense(bn.gradWeightAndBias.native) should be (
      Tools.dense(cloned.gradWeightAndBias.native))
  }

  "batch norm released" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val input = Tensor(100, 1, 10, 10).rand(-1, 1)
    val (channel, height, width) = (1, 10, 10)

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)
    val initCount = DnnStorage.get().count(!_._2)

    val bn = SpatialBatchNormalization(1, 0.0, initWeight = initWeight, initBias = initBias)

    val inputShape = Array(100, 1, 10, 10)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    bn.forward(input)
    val gradOutput = Tensor(inputShape).rand(-1, 1)
    bn.backward(input, gradOutput)

    bn.release()
    DnnStorage.get().count(_._2 == false) should be (initCount)
  }

  "batch norm with dense weights and gradients" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val input = Tensor(100, 1, 10, 10).rand(-1, 1)
    val gradOutput = Tensor(100, 1, 10, 10).rand(-1, 1)
    val (channel, height, width) = (1, 10, 10)

    val initWeight1 = Tensor(channel).rand(-1, 1)
    val initBias1 = Tensor(channel).fill(0)
    val initWeight2 = Tensor(channel).rand(-1, 1)
    val initBias2 = Tensor(channel).fill(0)

    val bn1 = SpatialBatchNormalization(1, 0.0, initWeight = initWeight1, initBias = initBias1)
    val bn2 = SpatialBatchNormalization(1, 0.0, initWeight = initWeight2, initBias = initBias2)

    val inputShape = Array(100, 1, 10, 10)
    for (bn <- List(bn1, bn2)) {
      bn.setRuntime(new MklDnnRuntime)
      bn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
      bn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
      bn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    }

    bn1.forward(input)
    bn1.backward(input, gradOutput)

    bn1.parameters()._1.zip(bn2.parameters()._1).foreach(x => x._1.copy(x._2))

    bn1.forward(input)
    bn1.backward(input, gradOutput)

    bn2.forward(input)
    bn2.backward(input, gradOutput)

    Tools.dense(bn1.output) should be (Tools.dense(bn2.output))
    Tools.dense(bn1.gradInput) should be (Tools.dense(bn2.gradInput))

    bn1.parameters()._2.zip(bn2.parameters()._2).foreach(x => x._1 should be (x._2))
  }

  "bn updateOutput" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val inputShape = Array(batchSize, channel, height, width)
    val defaultFormat = HeapData(inputShape, Memory.Format.nchw)
    val epsilon = 1e-5

    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initGradWPrimitives(Array(defaultFormat), TrainingPhase)

    val output = Tools.toNCHW(bn.forward(input).toTensor, bn.outputFormats()(0))

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)
    val nnOutput = nnBn.forward(input)

    Equivalent.nearequals(output, nnOutput) should be (true)
  }

  "bn updateOutput multi times" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val inputShape = Array(batchSize, channel, height, width)
    val defaultFormat = HeapData(inputShape, Memory.Format.nchw)
    val epsilon = 1e-5

    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initGradWPrimitives(Array(defaultFormat), TrainingPhase)

    TestUtils.manyTimes(bn.forward(input))(10)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    TestUtils.manyTimes(nnBn.forward(input))(10)

    val output = Tools.toNCHW(bn.output.toTensor, bn.outputFormats()(0))

    Equivalent.nearequals(output, nnBn.output.toTensor) should be (true)
  }

  "bn backward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val inputShape = Array(batchSize, channel, height, width)
    val defaultFormat = HeapData(inputShape, Memory.Format.nchw)
    val epsilon = 0.0f

    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resize(inputShape).rand(-1, 1)
    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initGradWPrimitives(Array(defaultFormat), TrainingPhase)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    bn.forward(input)
    nnBn.forward(input)

    val output = Tools.toNCHW(bn.output.toTensor, bn.outputFormats()(0))

    Equivalent.nearequals(output, nnBn.output) should be (true)

    bn.backward(input, gradOutput)
    val nnGradInput = nnBn.backward(input, gradOutput)

    val gradInput = Tools.toNCHW(bn.gradInput.toTensor, bn.gradInputFormats()(0))
    val weightAndBias = Tools.dense(bn.parameters()._2(0)).toTensor

    Equivalent.nearequals(gradInput, nnGradInput.toTensor) should be (true)
    Equivalent.nearequals(weightAndBias, nnBn.getParameters()._2) should be (true)
  }

//  "bn perf" should "work correctly" in {
//    // For PERF test. It seems sometimes batch norm maybe slower than java version.
//    val (batchSize, channel, height, width) = (4, 64, 112, 112)
//    val inputShape = Array(batchSize, channel, height, width)
//    val defaultFormat = HeapData(inputShape, Memory.Format.nChw8c)
//    val epsilon = 0.0f
//
//    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
//    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)
//
//    val initWeight = Tensor(channel).rand(-1, 1)
//    val initBias = Tensor(channel).rand(-1, 1)
//
//    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
//      initBias = initBias)
//    bn.setRuntime(new MklDnnRuntime)
//    bn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
//    bn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)
//    bn.initGradWPrimitives(Array(defaultFormat), TrainingPhase)
//
//    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
//      initWeight = initWeight, initBias = initBias)
//
//    val times = Utils.manyTimes {
//      bn.forward(input)
//      bn.backward(input, gradOutput)
//    } _
//
//    val nnTimes = Utils.manyTimes {
//      nnBn.forward(input)
//      nnBn.backward(input, gradOutput)
//    } _
//
//    times(10)
//    nnTimes(10)
//
//    val costs = times(50)._1
//    val nnCosts = nnTimes(50)._1
//
//    costs should be < (nnCosts)
//  }

  "a complicated batch norm" should "work correctly" in {
    val (channel, height, width) = (64, 112, 112)
    val epsilon = 1e-3
    val batchSize = 2

    RNG.setSeed(100)
    val input = Tensor[Float](Array(batchSize, 64, 112, 112)).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).copy(input)

    RNG.setSeed(100)
    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0f)

    val inputShape = input.size()
    val outputShape = input.size()
    val defaultFormat = HeapData(inputShape, Memory.Format.nchw)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)
    bn.initGradWPrimitives(Array(defaultFormat), TrainingPhase)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)

    bn.zeroGradParameters()
    nnBn.zeroGradParameters()

    val (weight, gradWeight) = bn.parameters()
    val (nnWeight, nnGradWeight) = nnBn.getParameters()
    Equivalent.nearequals(Tools.dense(weight(0)).toTensor, nnWeight) should be(true)
    Equivalent.nearequals(Tools.dense(gradWeight(0)).toTensor, nnGradWeight) should be(true)

    val out1 = bn.forward(input)
    val out2 = nnBn.forward(input)

    Equivalent.nearequals(Tools.dense(bn.output).toTensor, nnBn.output) should be (true)

    val gradInput = bn.backward(input, gradOutput)
    val nnGradInput = nnBn.backward(input, gradOutput)

    Equivalent.nearequals(Tools.dense(gradInput).toTensor, nnGradInput.toTensor,
      1e-3) should be (true)
    Equivalent.nearequals(Tools.dense(gradWeight(0)).toTensor, nnGradWeight, 1e-3) should be (true)
  }

  "A nChw8c input" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 256, 56, 56)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor(batchSize, channel, height, width).rand(-1, 1)

    val inputShape = input.size()
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nChw8c))
    val reorder2 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val dnn = Sequential()
      .add(reorder1)
      .add(SpatialBatchNormalization(channel, 1e-3, initWeight = initWeight, initBias = initBias))
      .add(reorder2)

    dnn.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    val blas = nn.Sequential().add(
        nn.SpatialBatchNormalization(channel, 1e-3, initWeight = initWeight, initBias = initBias))

    dnn.forward(input)
    blas.forward(input)

    dnn.backward(input, gradOutput)
    blas.backward(input, gradOutput)

    val gradWeight = Tools.dense(dnn.parameters()._2(0)).toTensor

    Equivalent.nearequals(dnn.output.toTensor, blas.output.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradWeight, blas.getParameters()._2, 1e-3) should be (true)
  }

//  "A nChw16c input" should "work correctly" in {
//    // only works on avx512 (SKX->)
//    val (batchSize, channel, height, width) = (2, 256, 56, 56)
//    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
//    val gradOutput = Tensor(batchSize, channel, height, width).rand(-1, 1)
//
//    val inputShape = input.size()
//    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nChw16c))
//    val reorder2 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
//
//    val initWeight = Tensor(channel).rand(-1, 1)
//    val initBias = Tensor(channel).rand(-1, 1)
//
//    val dnn = Sequential()
//      .add(reorder1)
//      .add(SpatialBatchNormalization(channel, 1e-3, initWeight = initWeight, initBias = initBias))
//      .add(reorder2)
//
//    dnn.compile(TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
//
//    val blas = nn.Sequential().add(
//        nn.SpatialBatchNormalization(channel, 1e-3, initWeight = initWeight, initBias = initBias))
//
//    dnn.forward(input)
//    blas.forward(input)
//
//    dnn.backward(input, gradOutput)
//    blas.backward(input, gradOutput)
//
//    val gradWeight = Tools.dense(dnn.parameters()._2(0)).toTensor
//
//    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor, 1e-4) should be (true)
//    DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor, 1e-4) should be (true)
//    DnnUtils.nearequals(gradWeight, blas.getParameters()._2, 1e-3) should be (true)
//  }

  "Sbn with relu fusion" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val shape = Array(batchSize, channel, height, width)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).fill(0)

    val bn1 = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val reorder1 = ReorderMemory(HeapData(shape, Memory.Format.nchw))
    val bn2 = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val reorder2 = ReorderMemory(HeapData(shape, Memory.Format.nchw))

    val model1 = Sequential().add(bn1).add(ReLU()).add(ReLU()).add(reorder1)
    model1.compile(TrainingPhase, Array(HeapData(shape, Memory.Format.nchw)))

    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "true")
    val model2 = Sequential().add(bn2).add(ReLU()).add(ReLU()).add(reorder2)
    model2.compile(TrainingPhase, Array(HeapData(shape, Memory.Format.nchw)))
    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "false")

    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)

    model1.forward(input)
    model2.forward(input)

    model1.output should be (model2.output)
  }

  "bn train and evaluate" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val input = Tensor(100, 1, 10, 10).fill(1.0f)
    val gradOutput = Tensor[Float]().resizeAs(input).fill(0.5f)
    val (channel, height, width) = (1, 10, 10)

    val initWeight = Tensor(channel).fill(0.3f)
    val initBias = Tensor(channel).fill(0)

    val bn = SpatialBatchNormalization(1, 1e-3, initWeight = initWeight, initBias = initBias)

    val runningMean = Tensor[Float](1).fill(1.0f)
    val runningVariance = Tensor[Float](1).fill(0.0f)

    val inputShape = Array(100, 1, 10, 10)
    bn.setRuntime(new MklDnnRuntime)
    bn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    bn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    bn.forward(input)
    bn.backward(input, gradOutput)
    bn.runningMean.dense should be (runningMean)
    bn.runningVariance.dense should be (runningVariance)

    bn.evaluate()
    bn.forward(input)

    bn.runningMean.dense should be (runningMean)
    bn.runningVariance.dense should be (runningVariance)
  }

  "a simple bach norm" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 2, 2)
    val shape = Array(batchSize, channel, height, width)
    val prototxt = s"""
         |name: "relu-simple"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "xavier"
         |    }
         |    shape: { dim: $batchSize dim: $channel dim: $height dim: $width }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "bn"
         |  name: "bn"
         |  type: "BatchNorm"
         |
         |  batch_norm_param {
         |    moving_average_fraction: 1.0
         |    filler { value: 1 }
         |    bias_filler { value: 1 }
         |    relu: false
         |    eps: 0.0
         |  }
         |}
       """.stripMargin

    val identity = Collect.run(prototxt)

    val input = Tools.getTensor("Fwrd_data", shape, identity)
    val output = Tools.getTensor("Fwrd_bn", shape, identity)
    val weight = Tools.getTensor("Fwrd_bn.Wght.3", Array(channel), identity)
    val bias = Tools.getTensor("Fwrd_bn.Wght.4", Array(channel), identity)
    val scale = Tools.getTensor("Fwrd_bn.Wght.2", Array(1), identity)
    val runningMean = Tools.getTensor("Fwrd_bn.Wght.0", Array(channel), identity)
    val runningVariance = Tools.getTensor("Fwrd_bn.Wght.1", Array(channel), identity)
    val gradOutput = Tools.getTensor("Bwrd_bn.loss", shape, identity)
    val gradInput = Tools.getTensor("Bwrd_bn", shape, identity)
    val gradWeight = Tools.getTensor("Bwrd_bn.Grad.3", Array(channel), identity)
    val gradBias = Tools.getTensor("Bwrd_bn.Grad.4", Array(channel), identity)

    val bn = new SpatialBatchNormalization(channel, eps = 0.0, momentum = 1.0,
      initWeight = weight, initBias = bias)

    val reorder1 = ReorderMemory(HeapData(shape, Memory.Format.nchw)).setName("reorder1")
    val reorder2 = ReorderMemory(HeapData(shape, Memory.Format.nchw)).setName("reorder2")
    val reorder3 = ReorderMemory(HeapData(shape, Memory.Format.nChw8c)).setName("reorder3")
    val reorder4 = ReorderMemory(HeapData(shape, Memory.Format.nchw)).setName("reorder4")

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(reorder3)
    seq.add(bn)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(shape, Memory.Format.nchw)))
    seq.reset()

    bn.zeroGradParameters()

    seq.forward(input)
    seq.backward(input, gradOutput)

    val weightAndBias = Tensor[Float](Array(2, channel))
    weightAndBias.select(1, 1).copy(weight)
    weightAndBias.select(1, 2).copy(bias)

    val gradWeightAndBias = Tensor[Float](Array(2, channel))
    gradWeightAndBias.select(1, 1).copy(gradWeight)
    gradWeightAndBias.select(1, 2).copy(gradBias)

    compare(weightAndBias.view(Array(2 * channel)), bn.weightAndBias.native)
    compare(output, seq.output)
    compare(runningMean, bn.runningMean.native)
    compare(runningVariance, bn.runningVariance.native)
    compare(gradWeightAndBias.view(Array(2 * channel)), bn.gradWeightAndBias.native)
    compare(gradInput, seq.gradInput)
  }

  "a simple bach norm inference" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val shape = Array(batchSize, channel, height, width)
    val prototxt = s"""
         |name: "relu-simple"
         |force_backward: true
         |state {
         |  phase: TEST
         |}
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "xavier"
         |    }
         |    shape: { dim: $batchSize dim: $channel dim: $height dim: $width }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "bn"
         |  name: "bn"
         |  type: "BatchNorm"
         |
         |  batch_norm_param {
         |    moving_average_fraction: 1.0
         |    filler { value: 1 }
         |    bias_filler { value: 0 }
         |    relu: false
         |    eps: 0.0
         |  }
         |
         |  phase: TEST
         |}
       """.stripMargin

    val identity = Collect.run(prototxt)

    val input = Tools.getTensor("Fwrd_data", shape, identity)
    val output = Tools.getTensor("Fwrd_bn", shape, identity)
    val weight = Tools.getTensor("Fwrd_bn.Wght.3", Array(channel), identity)
    val bias = Tools.getTensor("Fwrd_bn.Wght.4", Array(channel), identity)
    val scale = Tools.getTensor("Fwrd_bn.Wght.2", Array(1), identity)
    val runningMean = Tools.getTensor("Fwrd_bn.Wght.0", Array(channel), identity)
    val runningVariance = Tools.getTensor("Fwrd_bn.Wght.1", Array(channel), identity)

    val bn = new SpatialBatchNormalization(channel, eps = 0.0, momentum = 1.0,
      initWeight = weight, initBias = bias)
    bn.runningMean.copy(runningMean)
    bn.runningVariance.copy(runningVariance)

    val reorder1 = ReorderMemory(HeapData(shape, Memory.Format.nchw)).setName("reorder1")
    val reorder2 = ReorderMemory(HeapData(shape, Memory.Format.nchw)).setName("reorder2")

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(bn)
    seq.add(reorder2)
    seq.compile(Phase.InferencePhase, Array(HeapData(shape, Memory.Format.nchw)))
    seq.reset()
    seq.evaluate()

    seq.forward(input)

    val weightAndBias = Tensor[Float](Array(2, channel))
    weightAndBias.select(1, 1).copy(weight)
    weightAndBias.select(1, 2).copy(bias)

    compare(weightAndBias.view(Array(2 * channel)), bn.weightAndBias.native)
    compare(runningMean, bn.runningMean.native)
    compare(runningVariance, bn.runningVariance.native)

    val denseOutput = Tools.dense(bn.output).toTensor

    denseOutput.storage().array().zip(output.storage().array()).foreach { x =>
      if (x._2.isInfinity || x._2.isNaN)  x._1.isInfinity || x._1.isNaN should be (true)
    }
  }

  private def compare(src: Activity, dst: Activity): Unit = {
    if (src.isTensor) {
      Equivalent.nearequals(Tools.dense(src).toTensor, Tools.dense(dst).toTensor) should be (true)
    }
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
