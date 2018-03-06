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
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, SpatialConvolution}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ModelSpec extends FlatSpec with Matchers {

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

  "Inception_v1_dnn" should " be same with inception_v1" in {
    val batchSize = 2
    val (model1, batch1) = getModel("inception_v1", batchSize)
    val (model2, batch2) = getModel("inception_v1_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

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

  "Inception_v2_dnn" should " be same with inception_v2" in {
    val batchSize = 2
    val (model1, batch1) = getModel("inception_v2", batchSize)
    val (model2, batch2) = getModel("inception_v2_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

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
    // DnnUtils.nearequals(grad1, grad2)

    //    val (weight1, bias1) = model1.getParameters()
    //    val (weight2, bias2) = model2.getParameters()
    //
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-3) should be(true)


    println("done")
  }

  "Vgg16_dnn" should " be same with Vgg16_dnn" in {
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
    DnnUtils.nearequals(biasAll1, biasAll2, 1e-3) should be(true)

    println("done")
  }

  "Vgg19_dnn" should " be same with Vgg19_dnn" in {
    val batchSize = 2
    val (model1, batch1) = getModel("vgg19", batchSize)
    val (model2, batch2) = getModel("vgg19_dnn", batchSize)

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
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) shoul d be(true)
    DnnUtils.nearequals(biasAll1, biasAll2, 1e-3) should be(true)

    println("done")


  }
  "Resnet50-dnn" should " be same with resnet-50" in {
    val batchSize = 2
    val (model1, batch1) = getModel("resnet_50", batchSize)
    val (model2, batch2) = getModel("resnet_50_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

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
    // DnnUtils.nearequals(grad1, grad2)

    //    val (weight1, bias1) = model1.getParameters()
    //    val (weight2, bias2) = model2.getParameters()
    //
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)


    println("done")

  }
}
