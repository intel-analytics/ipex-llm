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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.{Xavier, Zeros}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnStorage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SpatialConvolutionSpec extends FlatSpec with Matchers {
  "ConvolutionDnn with format=nchw and ngroup=1" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    RNG.setSeed(100)
    val layer = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    conv.setRuntime(new MklDnnRuntime)
    conv.initFwdPrimitives(Array(HeapData(Array(2, 2, 23, 23), Memory.Format.nchw)), TrainingPhase)
    conv.initBwdPrimitives(Array(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)), TrainingPhase)
    conv.initGradWPrimitives(Array(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)), TrainingPhase)

    val output = Tools.toNCHW(conv.forward(input).toTensor, conv.outputFormats()(0))
    val grad1 = Tools.toNCHW(conv.updateGradInput(input, gradOutput).toTensor,
      conv.gradInputFormats()(0))
    conv.accGradParameters(input, gradOutput)

    val weight1 = Tools.toOIHW(conv.weight.native, conv.parametersWithShape()._1(0))
    val gradweight1 = Tools.toOIHW(conv.gradWeight.native, conv.parametersWithShape()._2(0))
    val bias1 = Tools.dense(conv.bias.native).toTensor[Float]
    val gradbias1 = Tools.dense(conv.gradBias.dense).toTensor

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = layer.bias
    val gradbias2 = layer.gradBias

    Equivalent.nearequals(weight1, weight2.resizeAs(weight1)) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2.resizeAs(gradweight1)) should be(true)
    Equivalent.nearequals(bias1, bias2) should be(true)
    Equivalent.nearequals(gradbias1, gradbias2) should be(true)
    Equivalent.nearequals(output.toTensor, output2) should be(true)
    Equivalent.nearequals(grad1.toTensor, grad2) should be(true)
  }

  "ConvolutionDnn with format=nchw and ngroup=2" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)

    conv.setRuntime(new MklDnnRuntime)
    conv.initFwdPrimitives(Array(HeapData(Array(2, 2, 23, 23), Memory.Format.nchw)), TrainingPhase)
    conv.initBwdPrimitives(Array(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)), TrainingPhase)
    conv.initGradWPrimitives(Array(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)), TrainingPhase)

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = layer.bias
    val gradbias2 = layer.gradBias

    val output = Tools.toNCHW(conv.forward(input).toTensor, conv.outputFormats()(0))
    val grad1 = Tools.toNCHW(conv.updateGradInput(input, gradOutput).toTensor,
      conv.gradInputFormats()(0))
    conv.accGradParameters(input, gradOutput)
    val weight1 = Tools.toOIHW(conv.weight.native, conv.parametersWithShape()._1(0))
    val gradweight1 = Tools.toOIHW(conv.gradWeight.native, conv.parametersWithShape()._2(0))
    val bias1 = Tools.dense(conv.bias.native).toTensor[Float]
    val gradbias1 = Tools.dense(conv.gradBias.native).toTensor[Float]

    Equivalent.nearequals(weight1, weight2) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2) should be(true)
    Equivalent.nearequals(bias1, bias2) should be(true)
    Equivalent.nearequals(gradbias1, gradbias2) should be(true)
    Equivalent.nearequals(output, output2) should be(true)
    Equivalent.nearequals(grad1, grad2) should be(true)
  }

  "ConvolutionDnn with relu " should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, ngroup)
    RNG.setSeed(100)
    val conv1 = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,
      ngroup)

    val relu = ReLU()
    val relu1 = nn.ReLU[Float](ip = false)

    val model = Sequential().add(conv).add(relu)
      .add(ReorderMemory(HeapData(Array(2, 4, 6, 6), Memory.Format.nchw)))
    model.compile(TrainingPhase, Array(HeapData(Array(2, 2, 23, 23), Memory.Format.nchw)))

    val model1 = nn.Sequential().add(conv1).add(relu1)

    model.forward(input)
    model.backward(input, gradOutput)

    model1.forward(input)
    model1.backward(input, gradOutput)

    val output = Tools.toNCHW(conv.output.toTensor, conv.outputFormats()(0))
    val gradInput = Tools.toNCHW(conv.gradInput.toTensor, conv.gradInputFormats()(0))

    val weight = Tools.toOIHW(conv.weight.native, conv.parametersWithShape()._1(0))
    val gradweight = Tools.toOIHW(conv.gradWeight.native, conv.parametersWithShape()._2(0))
    val bias = Tools.dense(conv.bias.native).toTensor
    val gradbias = Tools.dense(conv.gradBias.native).toTensor

    val output1 = conv1.output.toTensor
    val gradInput1 = conv1.gradInput

    val weight1 = conv1.weight
    val gradweight1 = conv1.gradWeight
    val bias1 = conv1.bias
    val gradbias1 = conv1.gradBias

    Equivalent.nearequals(weight, weight1) should be(true)
    Equivalent.nearequals(gradweight, gradweight1) should be(true)
    Equivalent.nearequals(bias, bias1) should be(true)
    Equivalent.nearequals(gradbias, gradbias1) should be(true)
    Equivalent.nearequals(output, output1) should be(true)
    Equivalent.nearequals(gradInput, gradInput1) should be(true)
  }

  "ConvolutionDnn with same params with vgg16" should "work correctly" in {
    val batchSize = 2
    val needPropagateBack: Boolean = true
    val inputShape = Array(batchSize, 3, 224, 224)
    val outputShape = Array(batchSize, 64, 112, 112)

    RNG.setSeed(100)
    val model1 = nn.SpatialConvolution[Float](3, 64, 7, 7, 2, 2, 3, 3, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
    model1.zeroGradParameters()
    val (weightAll1, gradWeightAll1) = model1.parameters()

    RNG.setSeed(100)
    val model2 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1)
    model2.zeroGradParameters()

    model2.setRuntime(new MklDnnRuntime)
    model2.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    model2.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    model2.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    val initWeight = Tools.fromOIHW(weightAll1(0), model2.parametersWithShape()._1(0))
    model2.weight.copy(initWeight)
    model2.bias.copy(model1.bias)

    RNG.setSeed(1)
    val input = Tensor(batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)
    val gradOutput = Tensor(outputShape).apply1(_ => RNG.uniform(0, 1).toFloat)

    val (weightAll2, gradWeightAll2) = model2.parameters()

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    var userOut2 = Tools.toNCHW(out2, model2.outputFormats()(0))

    Equivalent.nearequals(out1, userOut2, 1e-4) should be(true)

    val grad1 = model1.updateGradInput(input, gradOutput).toTensor[Float]
    val grad2 = model2.updateGradInput(input, gradOutput).toTensor[Float]

    val userGradInput2 = Tools.toNCHW(grad2, model2.gradInputFormats()(0))

    Equivalent.nearequals(grad1, userGradInput2, 1e-4) should be(true)

    model1.accGradParameters(input, gradOutput)
    model2.accGradParameters(input, gradOutput)

    val gw1 = model1.gradWeight
    val gb1 = model1.gradBias

    val gw2 = Tools.toOIHW(model2.gradWeight.native, model2.parametersWithShape()._2(0))
    val gb2 = Tools.dense(model2.gradBias.native).toTensor

    Equivalent.nearequals(gw1, gw2, 1e-4) should be(true)
    Equivalent.nearequals(gb1, gb2, 1e-3) should be(true)
  }

  "a simple convolution compared with caffe" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val txt = prototxt(inputShape, name, nOutput, kernel, pad, stride)

    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)
    conv.setRuntime(new MklDnnRuntime)
    conv.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    Tools.compare(txt, conv, inputShape, outputShape)
  }

  "conv exists some format conversion" should "work correctly" in {
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 112, 112)

    val name = "conv"
    val conv = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName(name)
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()

    val txt = prototxt(inputShape, name, outputShape(1), 7, 3, 2)
    val identity = Collect.run(txt)

    val input = Tools.getTensor("Fwrd_data", inputShape, identity)
    val gradOutput = Tools.getTensor(s"Bwrd_$name.loss", outputShape, identity)
    val output = Tools.getTensor(s"Fwrd_$name", outputShape, identity)
    val gradInput = Tools.getTensor(s"Bwrd_$name", inputShape, identity)

    if (conv.parameters() != null) {
      val params = conv.parameters()._1
      val infos = conv.parametersWithShape()._1
      val name = conv.getName()

      for (j <- params.indices) {
        val w = Tools.getTensor(s"Fwrd_$name.Wght.$j", params(j).size(), identity)
        params(j).copy(normal(w, infos(j)))
      }
    }

    seq.forward(input)
    seq.backward(input, gradOutput)

    Tools.compare2Tensors(Tools.dense(seq.output).toTensor, output) should be (true)
    Tools.compare2Tensors(Tools.dense(seq.gradInput).toTensor, gradInput) should be (true)

    val params = seq.parameters()._2
    val infos = conv.parametersWithShape()._2
    for (j <- params.indices) {
      val w = Tools.getTensor(s"Bwrd_$name.Grad.$j", params(j).size(), identity)
      Tools.compare2Tensors(params(j), normal(w, infos(j))) should be (true)
    }
  }

  "conv kernel 1x1 with reorder in container" should "work correctly" in {
    val inputShape = Array(4, 64, 56, 56)
    val outputShape = Array(4, 64, 56, 56)

    val name = "conv"
    val conv = SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0).setName(name)
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()

    val txt = prototxt(inputShape, name, outputShape(1), 1, 0, 1)
    val identity = Collect.run(txt)

    val input = Tools.getTensor("Fwrd_data", inputShape, identity)
    val gradOutput = Tools.getTensor(s"Bwrd_$name.loss", outputShape, identity)
    val output = Tools.getTensor(s"Fwrd_$name", outputShape, identity)
    val gradInput = Tools.getTensor(s"Bwrd_$name", inputShape, identity)

    if (conv.parameters() != null) {
      val params = conv.parameters()._1
      val infos = conv.parametersWithShape()._1
      val name = conv.getName()

      for (j <- params.indices) {
        val w = Tools.getTensor(s"Fwrd_$name.Wght.$j", params(j).size(), identity)
        params(j).copy(normal(w, infos(j)))
      }
    }

    seq.forward(input)
    seq.backward(input, gradOutput)

    Tools.compare2Tensors(Tools.dense(seq.output).toTensor, output) should be (true)
    Tools.compare2Tensors(Tools.dense(seq.gradInput).toTensor, gradInput) should be (true)

    val params = seq.parameters()._2
    val infos = conv.parametersWithShape()._2
    for (j <- params.indices.reverse) {
      val w = Tools.getTensor(s"Bwrd_$name.Grad.$j", params(j).size(), identity)
      Tools.compare2Tensors(params(j), normal(w, infos(j))) should be (true)
    }
  }

  "conv + bn" should "work correctly" in {
    val inputShape = Array(4, 3, 224, 224)
    val outputShape = Array(4, 64, 112, 112)
    val channel = 64

    val name = "conv"
    val conv = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName("conv")
    val bn = SpatialBatchNormalization(64, momentum = 1.0, eps = 100).setName("bn")
    // TODO we should insert a reorder manually
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw)).setName("reorder1")
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw)).setName("reorder2")

    val seq = Sequential()
    seq.add(reorder1)
    seq.add(conv)
    seq.add(bn)
    seq.add(reorder2)
    seq.compile(Phase.TrainingPhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    seq.reset()
    seq.training()

    val txt = prototxt2(inputShape, name, outputShape(1), 7, 3, 2) +
              """
                |layer {
                |  bottom: "conv"
                |  top: "bn"
                |  name: "bn"
                |  type: "BatchNorm"
                |
                |  batch_norm_param {
                |    moving_average_fraction: 1.0
                |    filler { value: 1 }
                |    bias_filler { value: 0 }
                |    relu: false
                |    eps: 100
                |  }
                |}
              """.stripMargin
    Tools.compare(txt, seq, inputShape, outputShape, 1e-2)
  }

  "conv serialized with java serialization method" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)
    conv.setRuntime(new MklDnnRuntime)
    conv.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    val input = Tensor(inputShape).rand(-1, 1)
    conv.forward(input)

    val cloned = SerializationUtils.clone(conv)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    cloned.forward(input)

    Tools.dense(conv.output) should be (Tools.dense(cloned.output))

    val gradOutput = Tensor(outputShape).rand(-1, 1)

    conv.backward(input, gradOutput)
    cloned.backward(input, gradOutput)

    Tools.dense(conv.gradInput) should be (Tools.dense(cloned.gradInput))
    Tools.dense(conv.gradWeight.native) should be (Tools.dense(cloned.gradWeight.native))
    Tools.dense(conv.gradBias.native) should be (Tools.dense(cloned.gradBias.native))
  }

  "conv release" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val name = "conv"
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val initCount = DnnStorage.get().count(!_._2)
    val conv = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1)
    conv.setName(name)
    conv.setRuntime(new MklDnnRuntime)
    conv.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    conv.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    val input = Tensor(inputShape).rand(-1, 1)
    val gradOutput = Tensor(outputShape).rand(-1, 1)
    conv.forward(input)
    conv.backward(input, gradOutput)

    conv.release()
    DnnStorage.get().count(_._2 == false) should be (initCount)
  }

  "conv with dense weights and gradients" should "work correctly" in {
    val inputShape = Array(4, 3, 5, 5)
    val outputShape = Array(4, 2, 3, 3)
    val nOutput = 2
    val kernel = 3
    val pad = 1
    val stride = 2

    val input = Tensor(inputShape).rand(-1, 1)
    val gradOutput = Tensor(outputShape).rand(-1, 1)

    val initWeight1 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initWeight2 = Tensor(Array(nOutput, inputShape(1), kernel, kernel)).rand(-1, 1)
    val initBias1 = Tensor(nOutput).rand(-1, 1)
    val initBias2 = Tensor(nOutput).rand(-1, 1)

    val conv1 = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1,
      initWeight = initWeight1, initBias = initBias1)
    conv1.setRuntime(new MklDnnRuntime)
    conv1.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    conv1.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    conv1.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    conv1.forward(input)
    conv1.backward(input, gradOutput)

    conv1.parameters()._1.zip(Array(initWeight2, initBias2)).foreach(x => x._1.copy(x._2))
    conv1.forward(input)
    conv1.backward(input, gradOutput)

    val conv2 = new SpatialConvolution(3, nOutput, kernel, kernel, stride, stride, pad, pad, 1,
      initWeight = initWeight2, initBias = initBias2)
    conv2.setRuntime(new MklDnnRuntime)
    conv2.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    conv2.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    conv2.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    conv2.forward(input)
    conv2.backward(input, gradOutput)

    Tools.dense(conv1.output) should be (Tools.dense(conv2.output))
    Tools.dense(conv1.gradInput) should be (Tools.dense(conv2.gradInput))

    conv1.parameters()._2.zip(conv2.parameters()._2).foreach(x => x._1 should be (x._2))
  }

  def prototxt(inputShape: Array[Int], name: String,
    nOutput: Int, kernel: Int, pad: Int, stride: Int): String = {
      s"""
         |name: "conv-simple"
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
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "conv"
         |  name: "$name"
         |  type: "Convolution"
         |  convolution_param {
         |    num_output: $nOutput
         |    kernel_size: $kernel
         |    pad: $pad
         |    stride: $stride
         |    weight_filler {
         |      type: "msra"
         |      variance_norm: FAN_OUT
         |    }
         |    bias_filler {
         |      type: "gaussian"
         |    }
         |  }
         |}
       """.stripMargin
  }

  def prototxt2(inputShape: Array[Int], name: String,
    nOutput: Int, kernel: Int, pad: Int, stride: Int): String = {
    s"""
       |name: "conv-simple"
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
       |      type: "uniform"
       |      min: -1000
       |      max: 1000
       |    }
       |    shape: { ${shape2Dim(inputShape)} }
       |  }
       |}
       |
         |layer {
       |  bottom: "data"
       |  top: "conv"
       |  name: "$name"
       |  type: "Convolution"
       |  convolution_param {
       |    num_output: $nOutput
       |    kernel_size: $kernel
       |    pad: $pad
       |    stride: $stride
       |    weight_filler {
       |      type: "msra"
       |      variance_norm: FAN_OUT
       |    }
       |    bias_filler {
       |      type: "gaussian"
       |    }
       |  }
       |}
       """.stripMargin
  }

  def normal(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = src.size().length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
    }

    if (defaultFormat != outputFormat.layout) {
      val inputFormat = HeapData(src.size(), defaultFormat)
      val reorder = ReorderMemory(inputFormat, outputFormat, null, null)
      reorder.setRuntime(new MklDnnRuntime)
      reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      reorder.updateOutput(src).toTensor
    } else {
      src
    }
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
