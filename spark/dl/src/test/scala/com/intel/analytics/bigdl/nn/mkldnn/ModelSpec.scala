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
import com.intel.analytics.bigdl.models.inception
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v1_NoAuxClassifier, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{VggForCifar10, Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{Module => _, _}
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
      case "inception_no_dnn" =>
        (Inception_v1_NoAuxClassifier_dnn(1000, false),
          MiniBatch(Tensor[Float](batchSize, 3, 224, 224).randn(),
            Tensor[Float](batchSize).fill(1)))
      case "inception_no" =>
        (Inception_v1_NoAuxClassifier(1000, false),
          MiniBatch(Tensor[Float](batchSize, 3, 224, 224).randn(),
            Tensor[Float](batchSize).fill(1)))
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
          "dataSet" -> DatasetType.ImageNet))
//        ResNet.shareGradInput(model)
//        ResNet.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))

      case "resnet_50_dnn" =>
        val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataSet" -> ResNet_dnn.DatasetType.ImageNet))
        //        ResNet_dnn.shareGradInput(model)
//        ResNet_dnn.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
    }
    _model.createDnnEngine(0)
    _model.createStream()
    (_model, input)
  }

  def formatEqual(out1: Tensor[Float], out2: Tensor[Float],
                  epsilion: Double, format: Int = 0): Unit = {
    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }

    DnnUtils.nearequals(out1, userOut2, epsilion) should be(true)
  }

  "inception_no_dnn" should "be same with inception_no" in {
    val batchSize = 2
    val (model1, batch1) = getModel("inception_no", batchSize)
    val (model2, batch2) = getModel("inception_no_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)

    val (weight1, gradWeight1) = model1.getParameters()
    val (weight2, gradWeight2) = model2.getParameters()
    DnnUtils.nearequals(weight1, weight2, 1e-10) should be(true)
    DnnUtils.nearequals(gradWeight1, gradWeight2, 1e-10) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    formatEqual(out1, out2, 1e-4)
    DnnUtils.nearequals(weight1, weight2, 1e-10) should be(true)
    DnnUtils.nearequals(gradWeight1, gradWeight2, 1e-10) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]

    // DnnUtils.nearequals(grad1, grad2, 1e-3) should be(true)
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.getunequals(gradWeight1, gradWeight2, 2e-4) should be(true)

    println("done")
  }

  "Inception_v1_dnn" should "be same with inception_v1" in {
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
    DnnUtils.nearequals(bias1, bias2, 2e-4) should be(true)

    println("done")

  }

  "Inception_v2_dnn" should "be same with inception_v2" in {
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

  "Vgg16_dnn" should "be same with Vgg16_dnn" in {
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
//    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
//    DnnUtils.nearequals(biasAll1, biasAll2, 1e-3) should be(true)

    val params1 = model1.parameters()._2
    val params2 = model2.parameters()._2

    params1.length should be (params2.length)

    for (i <- params1.indices.reverse) {
      DnnUtils.nearequals(params1(i), params2(i)) should be (true)
      println(s"module $i done")
    }
    DnnUtils.nearequals(model1.getParameters()._2, model2.getParameters()._2) should be (true)
    println("done")
  }

  "Vgg19_dnn" should "be same with Vgg19_dnn" in {
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
    DnnUtils.nearequals(weightAll1, weightAll2, 1e-4) should be(true)
    DnnUtils.nearequals(biasAll1, biasAll2, 1e-3) should be(true)

    println("done")


  }
  "Resnet50-dnn" should "be same with resnet-50" in {
    val batchSize = 2
    val (model1, batch1) = getModel("resnet_50", batchSize)
    val (model2, batch2) = getModel("resnet_50_dnn", batchSize)

    RNG.setSeed(1)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => RNG.uniform(0, 1).toFloat)

    val (weight1, bias1) = model1.getParameters()
    val (weight2, bias2) = model2.getParameters()

    DnnUtils.isEquals(weight1, weight2) should be(true)
    DnnUtils.isEquals(bias1, bias2) should be(true)

    val out1 = model1.forward(input).toTensor[Float]
    val out2 = model2.forward(input).toTensor[Float]

    if (out1.dim() < 4) {
      DnnUtils.nearequals(out1, out2, 1e-4) should be(true)
    } else {
      formatEqual(out1, out2, 1e-4)
    }

    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.nearequals(bias1, bias2, 1e-4) should be(true)

    val grad1 = model1.backward(input, out1).toTensor[Float]
    val grad2 = model2.backward(input, out1).toTensor[Float]
    // DnnUtils.nearequals(grad1, grad2)
    DnnUtils.nearequals(weight1, weight2, 1e-4) should be(true)
    DnnUtils.getunequals(bias1, bias2, 1e-2) should be(true)


    println("done")
  }

  "VGG on Cifar10" should "work correctly" in {
    RNG.setSeed(1)
    val dnn = VggForCifar10.dnn(10, hasDropout = false)
    val blas = VggForCifar10(10, hasDropout = false)

    dnn.getParameters()._1.copy(blas.getParameters()._1)
    dnn.getParameters()._1 should be (blas.getParameters()._1)

    val input = Tensor[Float](4, 3, 32, 32).rand(-1, 1)
    val gradOutput = Tensor[Float](4, 256, 4, 4).rand(-1, 1)

    for (i <- 0 until 1) {
      blas.forward(input)
      dnn.forward(input)
      blas.backward(input, gradOutput)
      dnn.backward(input, gradOutput)
      dnn.gradInput.toTensor[Float].storage()
    }

    DnnUtils.getunequals(blas.getParameters()._2, dnn.getParameters()._2, 1e-3) should be (true)

    println("done")
  }
}
