/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class InternalLayerNormSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "layerNorm" should "be able to generate correct result" in {
    val hiddenSize = 4
    val xValue = Tensor[Float](Array[Float](4f, 6f, 2f, 9f,
      3f, 5f, 7f, 4f,

      2f, 5f, 5f, 1f,
      4f, 3f, 6f, 4f), Array(2, 2, hiddenSize))
    val gradOValue = Tensor[Float](Array[Float](3f, 4f, 2f, 4f,
      4f, 1f, 2f, 2f,

      2f, 4f, 3f, 2f,
      4f, 1f, 3f, 1f), Array(2, 2, hiddenSize))

    val model = new InternalLayerNorm[Float](hiddenSize, 1e-5)

    val output = model.forward(xValue).toTensor[Float]
    val gradInput = model.backward(xValue, gradOValue).toTensor[Float]
    val gradients = model.parameters()._2

    val expectOutput = Tensor[Float](Array[Float](-0.48336786f, 0.2900207f, -1.2567564f,
      1.4501036f, -1.1832132f, 0.16903046f, 1.5212742f, -0.5070914f,
      -0.700139f, 0.98019457f, 0.98019457f, -1.2602501f,
      -0.22941478f, -1.1470739f, 1.6059034f, -0.22941478f), Array(2, 2, 4))
    val expectGradInput = Tensor[Float](Array[Float](0.043367267f, 0.20599616f,
      -0.119261384f, -0.13010204f,
      0.6761242f, -0.77271086f, 0.48294127f, -0.3863542f,
      -0.13179183f, 0.2965306f, -0.26358044f, 0.09884155f,
      1.7387214f, -0.48298407f, -0.24148142f, -1.014256f), Array(2, 2, 4))

    val expectGradient0 = Tensor[Float](Array[Float](-8.500894f, 4.1028175f, 8.287329f,
      2.0363169f), Array(1, 4))
    val expectGradient1 = Tensor[Float](Array[Float](13.0f, 10.0f, 10.0f, 9.0f), Array(1, 4))

    require(output.almostEqual(expectOutput, 1e-8))
    require(gradInput.almostEqual(expectGradInput, 1e-8))
    require(gradients.head.almostEqual(expectGradient0, 1e-8))
    require(gradients.last.almostEqual(expectGradient1, 1e-8))
  }

  "layerNorm" should "be able to generate correct result2" in {
    val hiddenSize = 4
    val xValue = Tensor[Float](Array[Float](4f, 6f, 2f, 9f,
      3f, 5f, 7f, 4f,

      2f, 5f, 5f, 1f,
      4f, 3f, 6f, 4f), Array(2, 2, hiddenSize))
    val gradOValue = Tensor[Float](Array[Float](3f, 4f, 2f, 4f,
      4f, 1f, 2f, 2f,

      2f, 4f, 3f, 2f,
      4f, 1f, 3f, 1f), Array(2, 2, hiddenSize))

    val model = new InternalLayerNorm[Float](hiddenSize, 1e-5)
    val w = model.parameters()._1
    w(0).copy(Tensor[Float](Array[Float](2.0f, 3.0f, 4.0f, 5.0f),
      Array(1, hiddenSize)))
    w(1).copy(Tensor[Float](Array[Float](2.5f, 3.5f, 4.5f, 5.5f),
      Array(1, hiddenSize)))

    val output = model.forward(xValue).toTensor[Float]
    val gradInput = model.backward(xValue, gradOValue).toTensor[Float]
    val gradients = model.parameters()._2

    val expectOutput = Tensor[Float](Array[Float](1.5332643f, 4.370062f,
      -0.5270257f, 12.750519f,
      0.13357353f, 4.0070915f, 10.585096f, 2.9645429f,
      1.099722f, 6.4405837f, 8.420778f, -0.80125046f,
      2.0411704f, 0.058778286f, 10.923614f, 4.3529263f), Array(2, 2, 4))
    val expectGradInput = Tensor[Float](Array[Float](-1.214294f, -0.35416794f,
      1.0191345f, 0.5493269f,
      0.13522625f, -2.8203938f, 0.9852042f, 1.6999645f,
      -2.2843778f, 0.28555012f, 0.28555012f, 1.7132769f,
      1.593823f, -0.2898159f, -0.14485359f, -1.1591544f), Array(2, 2, 4))

    val expectGradient0 = Tensor[Float](Array[Float](-8.500894f, 4.1028175f,
      8.287329f, 2.0363169f), Array(1, 4))
    val expectGradient1 = Tensor[Float](Array[Float](13.0f, 10.0f, 10.0f, 9.0f), Array(1, 4))

    require(output.almostEqual(expectOutput, 1e-8))
    require(gradInput.almostEqual(expectGradInput, 1e-8))
    require(gradients.head.almostEqual(expectGradient0, 1e-8))
    require(gradients.last.almostEqual(expectGradient1, 1e-8))
  }
}

class InternalLayerNormSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layerNorm = new InternalLayerNorm[Float](4).setName("layerNorm")
    val input = Tensor[Float](2, 2, 4).apply1(e => Random.nextFloat())
    runSerializationTest(layerNorm, input)
  }
}
