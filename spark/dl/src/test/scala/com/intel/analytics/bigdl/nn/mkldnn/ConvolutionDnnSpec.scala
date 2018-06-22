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
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.{ReLU => OReLU}
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ConvolutionDnnSpec extends FlatSpec with Matchers {

  "new conv " should "1" in {
    /* create data descriptors for convolution w/ no specified format */
//    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md;
//    CHECK(mkldnn_memory_desc_init(&conv_src_md, 4, conv_src_sizes, mkldnn_f32, mkldnn_any));
//    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, conv_weights_sizes, mkldnn_f32, mkldnn_any));
//    CHECK(mkldnn_memory_desc_init(&conv_bias_md, 1, conv_bias_sizes, mkldnn_f32, mkldnn_x));
//    CHECK(mkldnn_memory_desc_init(&conv_dst_md, 4, conv_dst_sizes, mkldnn_f32, mkldnn_any));
//
//    /* create a convolution */
//    mkldnn_convolution_desc_t conv_any_desc;
//    CHECK(mkldnn_convolution_forward_desc_init(
//      &conv_any_desc, mkldnn_forward, mkldnn_convolution_direct,
//      &conv_src_md, &conv_weights_md, &conv_bias_md, &conv_dst_md,
//      conv_strides, conv_padding, conv_padding, mkldnn_padding_zero));
//
//    mkldnn_primitive_desc_t conv_pd;
//    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc, engine, NULL));
//
//    const_mkldnn_primitive_desc_t weights_pd = mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_weights_pd, 0);
//    const mkldnn_memory_desc_t *wmd = mkldnn_primitive_desc_query_memory_d(weights_pd);
//    printf("fwd weight format %d ", (int)(wmd->format));

    val conv_src_sizes = Array(2, 3, 224, 224)
    val conv_weights_sizes = Array(64, 3, 3, 3)
    val conv_bias_sizes = Array(64)
    val conv_dst_sizes = Array(2, 64, 224, 224)
    val conv_strides = Array(1, 1)
    val conv_padding = Array(1, 1)

    val conv_src_md = MklDnnOps.memoryDescInit(4, conv_src_sizes, DataType.F32, Memory.Format.any)
    val conv_weights_md = MklDnnOps.memoryDescInit(4, conv_weights_sizes, DataType.F32, Memory.Format.any)
    val conv_bias_md = MklDnnOps.memoryDescInit(1, conv_bias_sizes, DataType.F32, Memory.Format.x)
    val conv_dst_md = MklDnnOps.memoryDescInit(4, conv_dst_sizes, DataType.F32, Memory.Format.any)

    val conv_any_desc = MklDnnOps.convForwardDescInit(
                    PropKind.Forward, AlgKind.ConvolutionDirect,
                    conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
                    conv_strides, conv_padding, conv_padding, MklDnn.PaddingKind.mkldnnPaddingZero)


    val weightBuffer = Tensor[Float](conv_weights_sizes).fill(1.0f)
    val biasBuffer = Tensor[Float](conv_bias_sizes).fill(1.0f)
    val inputBuffer = Tensor[Float](conv_src_sizes).fill(1.0f)
    val dstBuffer = Tensor[Float](conv_dst_sizes)


    val engine = Engine.Create(Engine.Kind.Cpu, 0)
    val conv_pd = MklDnnOps.primitiveDescCreate(conv_any_desc, engine, 0L)

    var internal_weightFormat = MklDnnOps.queryFormat(conv_pd, Query.WeightsPd)

    println("fwd weight format %d ", internal_weightFormat)
  }

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
    val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    RNG.setSeed(100)
    val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    val output = conv.forward(input)
    val grad1 = conv.updateGradInput(input, gradOutput)
    conv.accGradParameters(input, gradOutput)
//    conv.accGradParameters(input, gradOutput) // we will not add the original gradient now.
    val weight1 = conv.weight
    val gradweight1 = conv.gradWeight
    val bias1 = conv.bias
    val gradbias1 = conv.gradBias
    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
//    layer.accGradParameters(input, gradOutput)
    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = conv.bias
    val gradbias2 = conv.gradBias

    DnnUtils.nearequals(weight1, weight2) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
    DnnUtils.nearequals(bias1, bias2) should be(true)
    DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
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
    val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
//    layer.accGradParameters(input, gradOutput)
    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = conv.bias
    val gradbias2 = conv.gradBias

    val output = conv.forward(input)
    val grad1 = conv.updateGradInput(input, gradOutput)
    conv.accGradParameters(input, gradOutput)
//    conv.accGradParameters(input, gradOutput)
    val weight1 = conv.weight
    val gradweight1 = conv.gradWeight
    val bias1 = conv.bias
    val gradbias1 = conv.gradBias

    DnnUtils.nearequals(weight1, weight2) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
    DnnUtils.nearequals(bias1, bias2) should be(true)
    DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
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
    val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)
    val relu = ReLUDnn[Float](ip = false)
    val relu1 = OReLU[Float](ip = false)

    var output = conv.forward(input)
    relu.forward(output)
    val grad1 = relu.backward(output, gradOutput)
    val grad1_conv = conv.backward(input, grad1)

    val weight1 = conv.weight
    val gradweight1 = conv.gradWeight
    val bias1 = conv.bias
    val gradbias1 = conv.gradBias

    val output2 = layer.forward(input)
    relu1.forward(output2)
    val grad2 = relu1.backward(output2, gradOutput)
    val grad2_conv = layer.backward(input, grad2)

    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = conv.bias
    val gradbias2 = conv.gradBias

    DnnUtils.nearequals(weight1, weight2) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
    DnnUtils.nearequals(bias1, bias2) should be(true)
    DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
  }

  def getModel(module: String, batchSize: Int): (Module[Float], MiniBatch[Float]) = {
    RNG.setSeed(100)
    val (_model, input) = module match {
      case "inception_v1" =>
        (Inception_v1(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v1_dnn" =>
        (Inception_v1_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v2" =>
        (Inception_v2(1000), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v2_dnn" =>
        (Inception_v2_dnn(1000), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "vgg16" =>
        (Vgg_16(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg16_dnn" =>
        (Vgg_16_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg19" =>
        (Vgg_19(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg19_dnn" =>
        (Vgg_19_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "resnet_50" =>
        val model = ResNet(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> DatasetType.ImageNet))
        ResNet.shareGradInput(model)
        ResNet.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))

      case "resnet_50_dnn" =>
        val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> ResNet_dnn.DatasetType.ImageNet))
        //        ResNet_dnn.shareGradInput(model)
        //        ResNet_dnn.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
    }
    _model.createDnnEngine(0)
    _model.createStream()
    (_model, input)
  }

  "reorder to 25 " should "right" in {
    val weights_sizes = Array(64, 3, 3, 3)
    val weightBuffer = Tensor[Float](weights_sizes).rand()
    val outputBuffer = Tensor[Float](weights_sizes)
    val output2Buffer = Tensor[Float](weights_sizes)

    val inputFormat = Memory.Format.oihw
    val outputFormat = Memory.Format.Ohwi16o
    DnnUtils.reorderTwoTensor(weightBuffer, inputFormat, outputBuffer, outputFormat)
    DnnUtils.reorderTwoTensor(outputBuffer, outputFormat, output2Buffer, inputFormat)


    DnnUtils.nearequals(weightBuffer, output2Buffer) should be(true)

    println("done")
  }

  "reorder to 8 " should "right" in {
    val weights_sizes = Array(4, 1024, 1, 1)
    val weightBuffer = Tensor[Float](weights_sizes).rand()
    val outputBuffer = Tensor[Float](weights_sizes)
    val output2Buffer = Tensor[Float](weights_sizes)

    val inputFormat = Memory.Format.nchw
    val outputFormat = Memory.Format.nChw16c
    DnnUtils.reorderTwoTensor(weightBuffer, inputFormat, outputBuffer, outputFormat)
    DnnUtils.reorderTwoTensor(outputBuffer, outputFormat, output2Buffer, inputFormat)

    DnnUtils.nearequals(weightBuffer, output2Buffer) should be(true)
    DnnUtils.nearequals(weightBuffer, outputBuffer) should be(true)

    println("done")
  }

  "ConvolutionDnn with same params with vgg16" should "work correctly" in {
    val batchSize = 2
    val needPropagateBack: Boolean = true
    RNG.setSeed(100)
    val model1 = SpatialConvolution[Float](3, 64, 7, 7, 2, 2, 3, 3, 1,
      propagateBack = needPropagateBack).setInitMethod(weightInitMethod = Xavier, Zeros)

//    val model1 = SpatialConvolution[Float](3, 64, 3, 3, 1, 1, 1, 1,
//      propagateBack = needPropagateBack).setInitMethod(weightInitMethod = Xavier, Zeros)

    RNG.setSeed(100)
//    val model2 = ConvolutionDnnTest(3, 64, 3, 3, 1, 1, 1, 1,
//      propagateBack = needPropagateBack)
    val model2 = ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1,
      propagateBack = needPropagateBack).setInitMethod(weightInitMethod = Xavier, Zeros)
    model1.zeroGradParameters()
    model2.zeroGradParameters()

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)
    // val target = Tensor[Float](batchSize, 64, 224, 224).fill(1.0f)

//    model1.weight.fill(1.0f)
//    model2.weight.fill(1.0f)
//    model1.bias.fill(1.0f)
//    model2.bias.fill(1.0f)

//    model1.gradWeight.fill(0.1f)
//    model2.gradWeight.fill(0.1f)
//    model1.gradBias.fill(0.1f)
//    model2.gradBias.fill(0.1f)

    val (weightAll1, gradWeightAll1) = model1.getParameters()
    val (weightAll2, gradWeightAll2) = model2.getParameters()


    DnnUtils.nearequals(weightAll1, weightAll2) should be(true)
    DnnUtils.nearequals(gradWeightAll1, gradWeightAll2) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]
    // out2.storage()

    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }
    DnnUtils.nearequals(out1, userOut2, 1e-4) should be(true)
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
    DnnUtils.nearequals(gradWeightAll1, gradWeightAll2, 1e-3) should be(true)

    val grad1 = model1.updateGradInput(input, out1).toTensor[Float]
    val grad2 = model2.updateGradInput(input, out1).toTensor[Float]

    if (needPropagateBack) {
      grad2.storage()
      println("compare gradInput")
      DnnUtils.nearequals(grad1, grad2, 1e-4) should be(true)
    }
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
    DnnUtils.nearequals(gradWeightAll1, gradWeightAll2, 1e-3) should be(true)

    model1.accGradParameters(input, out1)
    model2.accGradParameters(input, out1)

    val gw1 = model1.gradWeight
    val gb1 = model1.gradBias

    val gw2 = model2.gradWeight
    val gb2 = model2.gradBias


    println("compare params")
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
    DnnUtils.nearequals(gradWeightAll1, gradWeightAll2, 1e-3) should be(true)

    println("done")
  }

  "test11" should "right" in {
    val batchSize = 2
    RNG.setSeed(100)
    val model1 = SpatialConvolution[Float](3, 64, 3, 3, 1, 1, 1, 1)
    RNG.setSeed(100)
    val model2 = ConvolutionDnn(3, 64, 3, 3, 1, 1, 1, 1)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

    model1.weight.fill(1.0f)
    model2.weight.fill(1.0f)


    model1.bias.fill(1.0f)
    model2.bias.fill(1.0f)

    val (weight1, gradweight1) = model1.getParameters()
    val (weight2, gradweight2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2, 1e-4) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }
    DnnUtils.nearequals(out1, userOut2, 1e-4) should be(true)


    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2, 1e-4) should be(true)

    out1.setPrimitiveDesc(0L)
    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2, 1e-4) should be(true)

    println("done")
  }


  "Inception_v1_dnn 111" should "be same with inception_v1" in {
    val batchSize = 2
    val (model1, batch1) = getModel("inception_v1", batchSize)
    val (model2, batch2) = getModel("inception_v1_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)

    val (weight1, bias1) = model1.getParameters()
    val (weight2, bias2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]
    DnnUtils.nearequals(out1, out2, 1e-4) should be(true)
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    println("done")

  }

  "Inception_v1_dnn" should "work correctly" in {
    val batchSize = 2
    val (model1, batch1) = getModel("inception_v1", batchSize)
    val (model2, batch2) = getModel("inception_v1_dnn", batchSize)

    val relu1 = OReLU[Float](true)
    val relu2 = ReLUDnn[Float](true)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)

    val (weight1, gradweight1) = model1.getParameters()
    val (weight2, gradweight2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2, 1e-4) should be(true)


    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

//    val reluout1 = relu1.forward(out1)
//    val reluout2 = relu2.forward(out2)
//    reluout2.storage()
//
//    var reluUser2 = Tensor[Float]()
//    if (reluout1.getFormat() != reluout2.getFormat() && reluout2.getFormat() != 5 && reluout2.getFormat() != 4) {
//      DnnUtils.reorderToUser(reluout2, reluUser2, 5)
//    } else {
//      reluUser2 = reluout2
//    }
//    DnnUtils.nearequals(reluout1, reluUser2, 1e-4) should be(true)
//
//    val gradOut = Tensor[Float]().resizeAs(out1)
//    RNG.setSeed(1)
//    gradOut.apply1(e => RNG.uniform(0, 1).toFloat)
//    val relugrad1 = relu1.backward(out1, gradOut).toTensor[Float]
//    val relugrad2 = relu2.backward(out2, gradOut).toTensor[Float]
//    relugrad2.storage()
//    var gradUser2 = Tensor[Float]()
//    if (relugrad1.getFormat() != relugrad2.getFormat() &&
//      relugrad2.getFormat() != 5 && relugrad2.getFormat() != 4) {
//      DnnUtils.reorderTwoTensor(relugrad2, 8, gradUser2, 5)
//    } else {
//      gradUser2 = relugrad2
//    }
//    DnnUtils.nearequals(relugrad1, gradUser2, 1e-4) should be(true)
//
//    println("done")
//
//
//    var userOut2 = Tensor[Float]()
//    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5 && out2.getFormat() != 4) {
//      DnnUtils.reorderToUser(out2, userOut2, 5)
//    } else {
//      userOut2 = out2
//    }
//    DnnUtils.nearequals(out1, userOut2, 1e-4) should be(true)

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2, 1e-4) should be(true)

    RNG.setSeed(1)
    out1.apply1(e => RNG.uniform(0, 1).toFloat)
    out1.setPrimitiveDesc(0L)


    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.getunequals(gradweight1, gradweight2, 1e-2) should be(true)

    println("done")
  }

//  "Inception_v2_dnn" should "work correctly" in {
//    val batchSize = 2
//    RNG.setSeed(1)
//    val (model1, batch1) = getModel("inception_v2", batchSize)
//    val (model2, batch2) = getModel("inception_v2_dnn", batchSize)
//
//    model1.reset()
//    model2.reset()
//
//    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)
//    val (weight1, bias1) = model1.getParameters()
//    val (weight2, bias2) = model2.getParameters()
//
//    weight2.copy(weight1)
//    bias2.copy(bias1)
//
//    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
//    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)
//
//    val out1 = model1.forward(input).toTensor[Float]
//    val out2 = model2.forward(input).toTensor[Float]
//    DnnUtils.nearequals(out1, out2, 1e-1) should be(true)
//    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
//    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)
//
//    val grad1 = model1.backward(input, out1).toTensor[Float]
//    val grad2 = model2.backward(input, out1).toTensor[Float]
//    // DnnUtils.nearequals(grad1, grad2)
//
//    //    val (weight1, bias1) = model1.getParameters()
//    //    val (weight2, bias2) = model2.getParameters()
//    //
//    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
//    DnnUtils.nearequals(bias1, bias2, 1e-3) should be(true)
//
//
//    println("done")
//
//  }

  "bn dnn" should "work correctly" in {
    val batchSize = 2
    RNG.setSeed(100)
    val model = SpatialConvolution[Float](3, 64, 7, 7, 2, 2, 3, 3, 1, false)
    val in = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)
    val input = Tensor[Float](100, 1, 10, 10).rand(-1, 1)
    val (channel, height, width) = (1, 10, 10)

    val initWeight = Tensor[Float](channel).rand(-1, 1)
    val initBias = Tensor[Float](channel).fill(0)

    val bn = mkldnn.SpatialBatchNormalization[Float](1, 0.0, initWeight = initWeight,
      initBias = initBias)
    val nnBn = nn.SpatialBatchNormalization[Float](1, 0.0, initWeight = initWeight,
      initBias = initBias)

    val out1 = bn.forward(input)
    val out2 = nnBn.forward(input)

    DnnUtils.nearequals(out1, out2, 1e-4) should be(true)


    val (weight1, gradweight1) = bn.getParameters()
    val (weight2, gradweight2) = nnBn.getParameters()

    val gradOutput = Tensor[Float]().resizeAs(input).rand()

    bn.backward(input, gradOutput)
    nnBn.backward(input, gradOutput)


    DnnUtils.nearequals(weight1, weight2)
    DnnUtils.nearequals(gradweight1, gradweight2)

    DnnUtils.nearequals(bn.output, nnBn.output)
    DnnUtils.nearequals(bn.gradInput, nnBn.gradInput)

    println("=" * 120)

  }

  "Inception_Layer_v2 dnn" should "work correctly" in {
    import com.intel.analytics.bigdl.models.inception
    val batchSize = 2
    RNG.setSeed(100)
    val model1 = inception.Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/")
    RNG.setSeed(100)
    val model2 = Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/")

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 256, 28, 28).randn()

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]
    out2.storage()

    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5 && out2.getFormat() != 4) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }
    DnnUtils.nearequals(out1, userOut2, 1e-4) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]
    // DnnUtils.nearequals(grad1, grad2)

    val (weight1, bias1) = model1.getParameters()
    val (weight2, bias2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)


    println("done")

  }

  "Inception_Layer_v1 dnn" should "work correctly" in {
    import com.intel.analytics.bigdl.models.inception
    val batchSize = 2
    RNG.setSeed(100)
    val model1 = inception.Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/")
    RNG.setSeed(100)
    val model2 = Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/")

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 192, 28, 28).randn()

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]
    out2.storage()

    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5 && out2.getFormat() != 4) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }
    DnnUtils.nearequals(out1, userOut2, 1e-4) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]
    var gradOut2 = Tensor[Float]()
    if (grad1.getFormat() != grad2.getFormat() && grad2.getFormat() != 5 && grad2.getFormat() != 4) {
      DnnUtils.reorderToUser(grad2, gradOut2, 5)
    } else {
      gradOut2 = grad2
    }
    DnnUtils.nearequals(grad1, gradOut2, 1e-4) should be(true)

    val (weight1, bias1) = model1.getParameters()
    val (weight2, bias2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)


    println("done")

  }

  "resnet_50_dnn" should "work correctly" in {
    val batchSize = 2
    // val (model1, batch1) = getModel("resnet_50", batchSize)
    // val (model2, batch2) = getModel("resnet_50_dnn", batchSize)

    val (model1, batch1) = getModel("vgg19", batchSize)
    val (model2, batch2) = getModel("vgg19_dnn", batchSize)


    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

    val (weight1, bias1) = model1.getParameters()
    val (weight2, bias2) = model2.getParameters()

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    out2.storage()
    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5 && out2.getFormat() != 4) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }
    DnnUtils.getunequals(out1, userOut2, 1e-3) should be(true)

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]
    // DnnUtils.nearequals(grad1, grad2)

    //    val (weight1, bias1) = model1.getParameters()
    //    val (weight2, bias2) = model2.getParameters()
    //
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.getunequals(bias1, bias2, 1e-3) should be(true)

    println("done")

  }

  "Vgg16_dnn 1111" should "be same with Vgg16_dnn" in {
    val batchSize = 2
    val (model1, batch1) = getModel("vgg16", batchSize)
    val (model2, batch2) = getModel("vgg16_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)

    val (weightAll1, biasAll1) = model1.getParameters()
    val (weightAll2, biasAll2) = model2.getParameters()

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]
    println("compare output")
    DnnUtils.nearequals(out1, out2, 1e-4) should be(true)

    val grad1 = model1.updateGradInput(input, out1).toTensor[Float]
    val grad2 = model2.updateGradInput(input, out1).toTensor[Float]
    grad1.storage()
    println("compare gradInput")
    DnnUtils.nearequals(grad1, grad2, 1e-4) should be(true)

    model1.accGradParameters(input, out1)
    model2.accGradParameters(input, out1)



    println("compare params")
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
    DnnUtils.getunequals(biasAll1, biasAll2, 1e-2) should be(true)

    println("done")
  }

  "Conv with fusion" should "work correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val batchSize = 2
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

    val conv = ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
    val conv2 = conv.cloneModule().asInstanceOf[ConvolutionDnn]
    val model = Sequential().add(conv2).add(ReLUDnn())

    model.evaluate()
    model.forward(input)

    conv.evaluate()
    conv.relu = true
    conv.forward(input)

    model.output.toTensor.storage()
    conv.output.storage()

    model.output.toTensor should be (conv.output)
  }

  "Conv Bn merge" should "work correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val batchSize = 2
    val input = Tensor[Float](batchSize, 1, 6, 6).rand(-1, 1)
    val dnn = Sequential()
      .add(ConvolutionDnn(1, 3, 2, 2, 2, 2, 1, 1, 1))
      .add(mkldnn.SpatialBatchNormalization[Float](3, eps = 1e-3))
    val merge = dnn.cloneModule() // .optimize()

    dnn.evaluate()
    dnn.forward(input)

    merge.evaluate()
    merge.optimize()
    merge.forward(input)

    dnn.output.toTensor.storage()
    merge.output.toTensor.storage()

    dnn.output should be (merge.output)
  }

  "Conv sum fusion" should "work correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat

    val input = Tensor[Float](2, 1, 6, 6).rand(0-1, 1)
    val conv = ConvolutionDnn(1, 3, 2, 2, 2, 2, 1, 1, 1)
    val conv2 = conv.cloneModule().asInstanceOf[mkldnn.ConvolutionDnn]
    val conv3 = conv.cloneModule().asInstanceOf[mkldnn.ConvolutionDnn]
    val conv4 = conv.cloneModule().asInstanceOf[mkldnn.ConvolutionDnn]

    conv.setRelu(false)
    conv.setSum(false)

    conv2.setRelu(false)
    conv2.setSum(true).setSumOp(conv)

    conv.forward(input)
    conv2.forward(input)
    conv2.output.storage()

    val caddTable = CAddTableDnn()
    val model2 = Sequential()
      .add(ConcatTableDnn()
        .add(conv3)
        .add(conv4))
      .add(caddTable)
      .add(ReLUDnn())
    model2.forward(input)
    caddTable.output.toTensor.asInstanceOf[MklDnnTensor[Float]].storage()

    conv2.output should be (caddTable.output)
  }
}
