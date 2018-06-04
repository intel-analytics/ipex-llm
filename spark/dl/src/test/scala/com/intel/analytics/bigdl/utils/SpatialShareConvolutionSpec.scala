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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.nn.{SpatialConvolution, SpatialShareConvolution}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.models.inception.{Inception_v1_NoAuxClassifier}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.{Vgg_16}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialShareConvolutionSpec extends FlatSpec with Matchers {
  val testSize = Array(
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 3, 1, 1, 1, 1, 1, 1, 1),
    (3, 1, 1, 1, 1, 1, 1, 1, 1),
    (3, 3, 1, 1, 1, 1, 1, 1, 1),
    (3, 6, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 2, 2, 1, 1, 1, 1, 1),
    (1, 3, 2, 2, 1, 1, 1, 1, 1),
    (3, 1, 2, 2, 1, 1, 1, 1, 1),
    (3, 3, 2, 2, 1, 1, 1, 1, 1),
    (3, 6, 2, 2, 1, 1, 1, 1, 1),
    (1, 1, 2, 2, 2, 2, 1, 1, 1),
    (1, 3, 2, 2, 2, 2, 1, 1, 1),
    (3, 1, 2, 2, 2, 2, 1, 1, 1),
    (3, 3, 2, 2, 2, 2, 1, 1, 1),
    (3, 6, 2, 2, 2, 2, 1, 1, 1),
    (1, 1, 2, 2, 2, 2, 0, 0, 1),
    (1, 3, 2, 2, 2, 2, 0, 0, 1),
    (3, 1, 2, 2, 2, 2, 0, 0, 1),
    (3, 3, 2, 2, 2, 2, 0, 0, 1),
    (3, 6, 2, 2, 2, 2, 0, 0, 1),
    (1, 1, 2, 2, 2, 2, 2, 2, 1),
    (1, 3, 2, 2, 2, 2, 2, 2, 1),
    (3, 1, 2, 2, 2, 2, 2, 2, 1),
    (3, 3, 2, 2, 2, 2, 2, 2, 1),
    (3, 6, 2, 2, 2, 2, 2, 2, 1),
    (3, 6, 2, 2, 2, 2, 2, 2, 3)
  )

  "SpatialSharedConvolution and SpatialConvolution" should "return the same result" in {
    RandomGenerator.RNG.setSeed(10)
    for (size <- testSize) {
      val conv = SpatialConvolution(
        size._1, size._2, size._3,
        size._4, size._5, size._6,
        size._7, size._8, size._9
      )
      val sharedConv = SpatialShareConvolution(
        size._1, size._2, size._3,
        size._4, size._5, size._6,
        size._7, size._8, size._9
      )
      sharedConv.getParameters()._1.copy(conv.getParameters()._1)

      val input = Tensor(1, size._1, 8, 8).rand()
      val output1 = conv.forward(input)
      val gradOutput = output1.clone().rand()
      val gradInput1 = conv.backward(input, gradOutput)

      val output2 = sharedConv.forward(input)
      val gradInput2 = sharedConv.backward(input, gradOutput)

      output1 should be (output2)
      gradInput1 should be (gradInput2)
      conv.gradWeight should be (sharedConv.gradWeight)
      conv.gradBias should be (sharedConv.gradBias)
    }
  }

  "SpatialSharedConvolution and SpatialConvolution without bias" should
      "return the same result" in {
    RandomGenerator.RNG.setSeed(10)
    for (size <- testSize) {
      val conv = SpatialConvolution(
        size._1, size._2, size._3,
        size._4, size._5, size._6,
        size._7, size._8, size._9, withBias = false
      )
      val sharedConv = SpatialShareConvolution(
        size._1, size._2, size._3,
        size._4, size._5, size._6,
        size._7, size._8, size._9, withBias = false
      )
      sharedConv.getParameters()._1.copy(conv.getParameters()._1)

      val input = Tensor(1, size._1, 8, 8).rand()
      val output1 = conv.forward(input)
      val gradOutput = output1.clone().rand()
      val gradInput1 = conv.backward(input, gradOutput)

      val output2 = sharedConv.forward(input)
      val gradInput2 = sharedConv.backward(input, gradOutput)

      output1 should be (output2)
      gradInput1 should be (gradInput2)
      conv.gradWeight should be (sharedConv.gradWeight)
      conv.gradBias should be (sharedConv.gradBias)
    }
  }

  "Inception" should "return right result" in {
    val inception = Inception_v1_NoAuxClassifier(1024)
    val sharedInception = SpatialShareConvolution.shareConvolution(
      inception.cloneModule())
    sharedInception.getParameters()._1.equals(
      inception.getParameters()._1) should be (true)

    Random.setSeed(100)
    val input = Tensor(4, 3, 224, 224).apply1(_ => Random.nextFloat())
    RandomGenerator.RNG.setSeed(100)
    val output1 = inception.forward(input).toTensor
    val gradOutput = output1.clone().apply1(_ => Random.nextFloat())
    val gradInput1 = inception.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(100)
    val output2 = sharedInception.forward(input).toTensor
    val gradInput2 = sharedInception.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)
    inception.getParameters()._2.equals(
      sharedInception.getParameters()._2) should be (true)
  }

  "Vgg_16" should "return right result" in {
    val vgg = Vgg_16(1000)
    val sharedVgg = SpatialShareConvolution.shareConvolution(
      vgg.cloneModule())
    sharedVgg.getParameters()._1.equals(
      vgg.getParameters()._1) should be (true)

    Random.setSeed(100)
    val input = Tensor(4, 3, 224, 224).apply1(_ => Random.nextFloat())
    RandomGenerator.RNG.setSeed(100)
    val output1 = vgg.forward(input).toTensor
    val gradOutput = output1.clone().apply1(_ => Random.nextFloat())
    val gradInput1 = vgg.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(100)
    val output2 = sharedVgg.forward(input).toTensor
    val gradInput2 = sharedVgg.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)
    vgg.getParameters()._2.equals(
      sharedVgg.getParameters()._2) should be (true)
  }

  "Resnet 18" should "return right result" in {
    val resnet = ResNet(1000, T("shortcutType" -> ShortcutType.B,
      "depth" -> 18, "dataSet" -> DatasetType.ImageNet))
    val sharedResnet = SpatialShareConvolution.shareConvolution(
      ResNet(1000, T("shortcutType" -> ShortcutType.B,
      "depth" -> 18, "dataSet" -> DatasetType.ImageNet)))
    sharedResnet.getParameters()._1.copy(resnet.getParameters()._1)

    Random.setSeed(100)
    val input = Tensor(4, 3, 224, 224).apply1(_ => Random.nextFloat())
    RandomGenerator.RNG.setSeed(100)
    val output1 = resnet.forward(input).toTensor
    val gradOutput = output1.clone().apply1(_ => Random.nextFloat())
    val gradInput1 = resnet.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(100)
    val output2 = sharedResnet.forward(input).toTensor
    val gradInput2 = sharedResnet.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)
    resnet.getParameters()._2 should be (sharedResnet.getParameters()._2)
  }

}
